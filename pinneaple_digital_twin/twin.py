"""Core DigitalTwin class.

A DigitalTwin wraps a trained surrogate model (PINN, GNN, DeepONet, etc.),
manages real-time data streams, performs data assimilation, detects
anomalies, and maintains a live state estimate of the physical system.

Architecture
------------
    Sensors/Streams  -->  ObsQueue  -->  update_loop()
                                             |
                                    assimilation (EKF/EnKF)
                                             |
                                    model prediction
                                             |
                                    anomaly monitoring
                                             |
                                    SystemState (current estimate)

Usage
-----
>>> from pinneaple_digital_twin import DigitalTwin
>>> from pinneaple_digital_twin.io import MockStream
>>> import torch

>>> # Load a trained model
>>> model = torch.load("my_pinn.pt")

>>> # Create digital twin
>>> dt = DigitalTwin(model, field_names=["u", "v", "p"])

>>> # Add a mock sensor stream
>>> stream = MockStream(
...     sensor_id="inlet_probe",
...     field_names=["u", "v"],
...     generator_fn=lambda t: {"u": 1.0 + 0.05 * np.sin(t), "v": 0.0},
...     coords={"x": 0.0, "y": 0.5},
... )
>>> dt.add_stream(stream)
>>> dt.start()        # begin update loop
>>> # ... use dt.state to access current estimate
>>> dt.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state import Observation, SystemState
from .io.sensors import SensorRegistry
from .io.stream import BaseStream
from .monitoring.anomaly import AnomalyMonitor, ThresholdDetector, ZScoreDetector

logger = logging.getLogger(__name__)

# Optional torch import for model inference
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DigitalTwinConfig:
    """Configuration for the DigitalTwin."""

    # Inference
    device: str = "cuda" if (_TORCH_AVAILABLE and __import__("torch").cuda.is_available()) else "cpu"
    batch_size: int = 4096        # points per inference batch (GPU paging)
    use_amp: bool = False         # automatic mixed precision for inference

    # Update loop
    update_interval: float = 0.5  # seconds between model inference passes
    obs_queue_maxsize: int = 1000 # max buffered observations

    # History
    max_history: int = 500        # max state snapshots to keep in RAM

    # Data assimilation
    assimilation: str = "none"    # "none" | "ekf" | "enkf"
    enkf_n_ens: int = 100
    enkf_inflation: float = 1.02

    # Anomaly detection
    anomaly_z_threshold: float = 3.0
    anomaly_window: int = 100

    # Online re-training (optional)
    enable_online_retraining: bool = False
    retrain_every_n_steps: int = 100
    retrain_lr: float = 1e-5
    retrain_steps: int = 10


# ---------------------------------------------------------------------------
# DigitalTwin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """
    Digital twin that fuses a trained surrogate model with live sensor data.

    Parameters
    ----------
    model : nn.Module or callable
        Trained surrogate model.  Called as ``model(x) -> predictions``.
        If the model returns a dict, ``field_names`` keys are extracted.
    field_names : list[str]
        Names of the fields the model predicts (e.g. ["u", "v", "p"]).
    coord_names : list[str]
        Names of the coordinate inputs (e.g. ["x", "y"]).
    config : DigitalTwinConfig
        Configuration dataclass.
    assimilation_filter : optional
        Pre-configured EKF/EnKF instance; if None and config.assimilation
        is set, a default filter will be built automatically.
    """

    def __init__(
        self,
        model: Any,
        field_names: List[str],
        coord_names: Optional[List[str]] = None,
        *,
        config: Optional[DigitalTwinConfig] = None,
        assimilation_filter: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.field_names = list(field_names)
        self.coord_names = list(coord_names) if coord_names else ["x", "y"]
        self.cfg = config or DigitalTwinConfig()

        # Device / AMP setup
        if _TORCH_AVAILABLE:
            self.device = torch.device(self.cfg.device)
            if hasattr(model, "to"):
                model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
        else:
            self.device = None

        # Core state
        self.state: SystemState = SystemState(
            fields={f: np.array([], dtype=np.float32) for f in self.field_names},
            coords={c: np.array([], dtype=np.float32) for c in self.coord_names},
        )

        # Streams and queues
        self._streams: List[BaseStream] = []
        self._obs_queue: queue.Queue = queue.Queue(
            maxsize=self.cfg.obs_queue_maxsize
        )
        self.sensor_registry = SensorRegistry()

        # Assimilation filter
        self._filter = assimilation_filter

        # Anomaly monitoring
        self.anomaly_monitor = AnomalyMonitor()
        self.anomaly_monitor.add_detector(
            ZScoreDetector(
                z_threshold=self.cfg.anomaly_z_threshold,
                window_size=self.cfg.anomaly_window,
            )
        )

        # Callbacks
        self._obs_callbacks: List[Callable[[Observation], None]] = []
        self._state_callbacks: List[Callable[[SystemState], None]] = []
        self._anomaly_callbacks: List[Callable[[Any], None]] = []

        # Internal bookkeeping
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._step_count: int = 0
        self._last_observations: List[Observation] = []

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def add_stream(self, stream: BaseStream) -> None:
        """Register a data stream."""
        stream.attach_queue(self._obs_queue)
        self._streams.append(stream)

    def remove_stream(self, sensor_id: str) -> None:
        """Stop and remove the stream for a given sensor_id."""
        to_remove = [s for s in self._streams if s.sensor_id == sensor_id]
        for s in to_remove:
            s.stop()
            self._streams.remove(s)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_observation(self, fn: Callable[[Observation], None]) -> None:
        """Register a callback invoked on every incoming observation."""
        self._obs_callbacks.append(fn)

    def on_state_update(self, fn: Callable[[SystemState], None]) -> None:
        """Register a callback invoked after each state update."""
        self._state_callbacks.append(fn)

    def on_anomaly(self, fn: Callable[[Any], None]) -> None:
        """Register a callback invoked when an anomaly is detected."""
        self._anomaly_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Model inference helpers
    # ------------------------------------------------------------------

    def _predict_at_coords(
        self,
        coords: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run the surrogate model on the given coordinates.

        coords : {name: array of shape (N,)}
        returns: {field_name: array of shape (N,)}
        """
        if not coords:
            return {}

        n = next(iter(coords.values())).shape[0]
        col_order = self.coord_names
        X_np = np.column_stack([coords.get(c, np.zeros(n)) for c in col_order]).astype(np.float32)

        if _TORCH_AVAILABLE and hasattr(self.model, "parameters"):
            return self._torch_predict(X_np)
        else:
            # numpy/callable model
            out = self.model(X_np)
            if isinstance(out, dict):
                return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 1:
                out = out[:, None]
            return {f: out[:, i] for i, f in enumerate(self.field_names) if i < out.shape[1]}

    def _torch_predict(self, X_np: "np.ndarray") -> Dict[str, "np.ndarray"]:
        """Batched GPU inference with optional AMP."""
        import torch
        results: Dict[str, List["np.ndarray"]] = {f: [] for f in self.field_names}
        N = X_np.shape[0]
        bs = self.cfg.batch_size

        with torch.no_grad():
            for start in range(0, N, bs):
                chunk = torch.from_numpy(X_np[start: start + bs]).to(self.device)
                if self.cfg.use_amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = self.model(chunk)
                else:
                    out = self.model(chunk)

                # Unwrap model output
                if hasattr(out, "y"):          # PINNOutput / OperatorOutput
                    out = out.y
                if isinstance(out, dict):
                    for f in self.field_names:
                        if f in out:
                            results[f].append(out[f].cpu().float().numpy())
                    continue
                out_np = out.cpu().float().numpy()
                if out_np.ndim == 1:
                    out_np = out_np[:, None]
                for i, f in enumerate(self.field_names):
                    if i < out_np.shape[1]:
                        results[f].append(out_np[:, i])

        return {f: np.concatenate(arrs, axis=0) for f, arrs in results.items() if arrs}

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _process_observation(self, obs: Observation) -> None:
        """Calibrate and process one incoming observation."""
        for cb in self._obs_callbacks:
            try:
                cb(obs)
            except Exception as exc:
                logger.warning(f"obs callback error: {exc}")

        self._last_observations.append(obs)
        # Keep only recent observations
        self._last_observations = self._last_observations[-200:]

    def _drain_queue(self) -> List[Observation]:
        """Drain all pending observations from the queue."""
        obs_list: List[Observation] = []
        while True:
            try:
                obs = self._obs_queue.get_nowait()
                self._process_observation(obs)
                obs_list.append(obs)
            except queue.Empty:
                break
        return obs_list

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_state(self, new_fields: Dict[str, np.ndarray]) -> None:
        """Apply model predictions to the current system state."""
        self.state.timestamp = time.time()
        for k, v in new_fields.items():
            self.state.fields[k] = v

    def _check_anomalies(
        self,
        observations: List[Observation],
        predictions: Dict[str, np.ndarray],
    ) -> None:
        """Compare point predictions against observations."""
        for obs in observations:
            if obs.coords is None:
                continue
            # Find predicted value at the observation location
            # (nearest-point lookup for simplicity)
            pred_vals: Dict[str, float] = {}
            for f in obs.values:
                if f in predictions:
                    # Single-point prediction at the sensor location
                    single_coords = {c: np.array([obs.coords.get(c, 0.0)]) for c in self.coord_names}
                    try:
                        single_pred = self._predict_at_coords(single_coords)
                        if f in single_pred:
                            pred_vals[f] = float(single_pred[f][0])
                    except Exception:
                        pass

            if pred_vals:
                events = self.anomaly_monitor.check(
                    obs.timestamp, obs.sensor_id, obs.values, pred_vals
                )
                for ev in events:
                    for cb in self._anomaly_callbacks:
                        try:
                            cb(ev)
                        except Exception as exc:
                            logger.warning(f"anomaly callback error: {exc}")

    def _assimilation_update(
        self,
        observations: List[Observation],
    ) -> None:
        """Run data assimilation filter with new observations."""
        if self._filter is None or not observations:
            return
        try:
            # Build observation vector from latest observations
            y_vals = []
            for obs in observations[-1:]:  # Use the most recent obs
                for f in self.field_names:
                    if f in obs.values:
                        y_vals.append(obs.values[f])
            if y_vals:
                y = np.array(y_vals, dtype=np.float64)
                result = self._filter.step(y)
                # Update state residuals
                if "innovation" in result:
                    inn = result["innovation"]
                    for i, f in enumerate(self.field_names):
                        if i < len(inn):
                            self.state.residuals[f] = float(inn[i])
                if "P" in result:
                    self.state.covariance = result["P"]
        except Exception as exc:
            logger.warning(f"Assimilation error: {exc}")

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _run_update_loop(self) -> None:
        """Background thread: drain queue -> predict -> update state."""
        while self._running:
            t0 = time.time()
            try:
                observations = self._drain_queue()

                # Run model on current state coordinates (if they exist)
                if self.state.coords and any(v.size > 0 for v in self.state.coords.values()):
                    new_fields = self._predict_at_coords(self.state.coords)
                    self._update_state(new_fields)

                # Data assimilation
                if observations:
                    self._assimilation_update(observations)
                    self._check_anomalies(observations, self.state.fields)

                # Persist snapshot to history
                self.state.push_to_history(max_history=self.cfg.max_history)
                self.state.metadata["step"] = self._step_count
                self._step_count += 1

                # State callbacks
                for cb in self._state_callbacks:
                    try:
                        cb(self.state)
                    except Exception as exc:
                        logger.warning(f"state callback error: {exc}")

            except Exception as exc:
                logger.error(f"Update loop error: {exc}", exc_info=True)

            elapsed = time.time() - t0
            sleep = max(0.0, self.cfg.update_interval - elapsed)
            time.sleep(sleep)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all data streams and the update loop."""
        if self._running:
            logger.warning("DigitalTwin is already running.")
            return
        self._running = True
        for stream in self._streams:
            stream.start(self._obs_queue)
        self._update_thread = threading.Thread(
            target=self._run_update_loop, daemon=True, name="dt-update-loop"
        )
        self._update_thread.start()
        logger.info("DigitalTwin started.")

    def stop(self) -> None:
        """Stop the update loop and all streams."""
        self._running = False
        for stream in self._streams:
            stream.stop()
        if self._update_thread is not None:
            self._update_thread.join(timeout=10.0)
        logger.info("DigitalTwin stopped.")

    def __enter__(self) -> "DigitalTwin":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Snapshot / forecast helpers
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return the current state as a JSON-serialisable dict."""
        return self.state.snapshot()

    def predict(
        self,
        coords: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run a one-shot prediction at arbitrary coordinates.

        coords : {coord_name: array (N,)}
        returns: {field_name: array (N,)}
        """
        return self._predict_at_coords(coords)

    def set_domain_coords(
        self,
        coords: Dict[str, np.ndarray],
    ) -> None:
        """
        Set the coordinate grid on which the model is evaluated
        at each update-loop tick.

        coords : {coord_name: array (N,)}
        """
        for k, v in coords.items():
            self.state.coords[k] = np.asarray(v, dtype=np.float32)
        # Initialise field arrays to correct size
        n = next(iter(coords.values())).shape[0]
        for f in self.field_names:
            if f not in self.state.fields or self.state.fields[f].shape != (n,):
                self.state.fields[f] = np.zeros(n, dtype=np.float32)

    def get_history_df(self) -> "Any":
        """
        Return history as a pandas DataFrame (if pandas is available).

        Columns: timestamp + all field means.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        rows = []
        for snap in self.state.history:
            row = {"timestamp": snap["timestamp"]}
            for f, vals in snap["fields"].items():
                arr = np.asarray(vals)
                if arr.size > 0:
                    row[f"mean_{f}"] = float(arr.mean())
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Builder convenience function
# ---------------------------------------------------------------------------

def build_digital_twin(
    model: Any,
    field_names: List[str],
    *,
    coord_names: Optional[List[str]] = None,
    device: str = "cpu",
    update_interval: float = 0.5,
    anomaly_z_threshold: float = 3.0,
    assimilation: str = "none",
    use_amp: bool = False,
) -> DigitalTwin:
    """
    Convenience factory for DigitalTwin.

    Parameters
    ----------
    model          Trained surrogate model (nn.Module or callable).
    field_names    List of physical field names the model predicts.
    coord_names    Coordinate input names (default ["x","y"]).
    device         "cpu" | "cuda" | "cuda:0" etc.
    update_interval  seconds between update-loop ticks.
    anomaly_z_threshold  Z-score threshold for anomaly detection.
    assimilation   "none" | "ekf" | "enkf".
    use_amp        Use mixed precision for GPU inference.
    """
    cfg = DigitalTwinConfig(
        device=device,
        update_interval=update_interval,
        anomaly_z_threshold=anomaly_z_threshold,
        assimilation=assimilation,
        use_amp=use_amp,
    )
    return DigitalTwin(model, field_names, coord_names=coord_names, config=cfg)
