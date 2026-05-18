"""Real-time inference pipeline for digital twins.

Enforces the hard constraint:  T_inference ≤ T_data_input

When a model takes longer to run than the data arrival period the pipeline
can:
  - ``"warn"``    — log a latency warning but continue
  - ``"skip"``    — drop the current frame and keep up
  - ``"throttle"``— reduce the inference grid to bring latency back under budget
  - ``"raise"``   — raise LatencyBudgetExceeded

Throughput and latency are tracked via an exponential moving average.

Usage
-----
>>> from pinneapple_systems.digital_twin.realtime_pipeline import RealtimePipeline
>>> pipeline = RealtimePipeline(model, data_period=0.1, field_names=["u", "v"])
>>> pipeline.start()
>>> pipeline.push({"u": 1.2, "v": 0.0}, timestamp=time.time())
>>> state = pipeline.latest_state()
>>> pipeline.stop()
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LatencyBudgetExceeded(RuntimeError):
    """Raised when inference time exceeds data_period and policy='raise'."""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Rolling latency statistics."""
    n_frames: int = 0
    n_skipped: int = 0
    n_budget_violations: int = 0
    mean_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    budget_ms: float = 0.0          # data_period in ms
    utilisation: float = 0.0        # mean_inference / budget (0–1+)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_skipped": self.n_skipped,
            "n_violations": self.n_budget_violations,
            "mean_inference_ms": round(self.mean_inference_ms, 3),
            "max_inference_ms": round(self.max_inference_ms, 3),
            "budget_ms": round(self.budget_ms, 3),
            "utilisation": round(self.utilisation, 4),
        }


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class _RingBuffer:
    """Thread-safe fixed-length ring buffer for incoming observations."""

    def __init__(self, maxlen: int = 512) -> None:
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, item: Any) -> None:
        with self._lock:
            self._buf.append(item)

    def drain(self) -> List[Any]:
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def peek_last(self) -> Optional[Any]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ---------------------------------------------------------------------------
# Latency guard
# ---------------------------------------------------------------------------

class _LatencyGuard:
    """
    Tracks inference latency with EMA and enforces the time budget.

    alpha : EMA smoothing factor (0 = no smoothing, 1 = ignore history)
    """

    def __init__(self, budget_s: float, alpha: float = 0.1) -> None:
        self.budget_s = float(budget_s)
        self.alpha = float(alpha)
        self._ema_s: float = 0.0
        self._max_s: float = 0.0
        self._n: int = 0

    def record(self, elapsed_s: float) -> None:
        if self._n == 0:
            self._ema_s = elapsed_s
        else:
            self._ema_s = (1 - self.alpha) * self._ema_s + self.alpha * elapsed_s
        self._max_s = max(self._max_s, elapsed_s)
        self._n += 1

    @property
    def over_budget(self) -> bool:
        return self._ema_s > self.budget_s

    @property
    def utilisation(self) -> float:
        return self._ema_s / (self.budget_s + 1e-12)

    def reset_max(self) -> None:
        self._max_s = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "mean_inference_ms": self._ema_s * 1000.0,
            "max_inference_ms": self._max_s * 1000.0,
            "budget_ms": self.budget_s * 1000.0,
            "utilisation": self.utilisation,
        }


# ---------------------------------------------------------------------------
# RealtimePipeline
# ---------------------------------------------------------------------------

@dataclass
class RealtimePipelineConfig:
    """Configuration for RealtimePipeline."""
    data_period: float = 0.1         # expected seconds between data inputs
    latency_policy: str = "warn"     # "warn" | "skip" | "throttle" | "raise"
    buffer_maxlen: int = 512         # ring buffer capacity
    latency_alpha: float = 0.1       # EMA smoothing for latency tracking
    throttle_min_points: int = 64    # minimum grid points when throttling
    throttle_step: float = 0.5       # fraction of points to drop on throttle
    device: str = "cpu"
    use_amp: bool = False


class RealtimePipeline:
    """
    Real-time inference pipeline wrapping a surrogate model.

    Guarantees T_inference ≤ T_data_input via the chosen ``latency_policy``.

    Parameters
    ----------
    model        : trained PyTorch model (callable: tensor -> tensor or Output)
    data_period  : seconds between data arrivals (sets the time budget)
    field_names  : names of fields the model predicts
    coord_names  : names of spatial inputs
    config       : full config dataclass
    on_state     : optional callback invoked with the latest state dict after
                   each successful inference pass
    on_latency_violation : optional callback invoked when EMA > budget
    """

    def __init__(
        self,
        model: Any,
        data_period: float,
        field_names: List[str],
        coord_names: Optional[List[str]] = None,
        *,
        config: Optional[RealtimePipelineConfig] = None,
        on_state: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_latency_violation: Optional[Callable[[LatencyStats], None]] = None,
    ) -> None:
        self.model = model
        self.field_names = list(field_names)
        self.coord_names = list(coord_names or ["x", "y"])

        self.cfg = config or RealtimePipelineConfig(data_period=data_period)
        self.cfg.data_period = float(data_period)

        self._guard = _LatencyGuard(
            budget_s=self.cfg.data_period,
            alpha=self.cfg.latency_alpha,
        )
        self._buffer = _RingBuffer(maxlen=self.cfg.buffer_maxlen)
        self._state_queue: queue.Queue = queue.Queue(maxsize=1)

        self._on_state = on_state
        self._on_violation = on_latency_violation

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._n_frames = 0
        self._n_skipped = 0
        self._n_violations = 0

        # Domain coordinates used for field inference (set via set_domain_coords)
        self._domain_coords: Optional[np.ndarray] = None  # (N, D)
        self._throttle_mask: Optional[np.ndarray] = None  # boolean (N,)

        try:
            import torch
            self._torch = torch
            self._device = torch.device(self.cfg.device)
            if hasattr(model, "to"):
                model.to(self._device)
            if hasattr(model, "eval"):
                model.eval()
        except ImportError:
            self._torch = None
            self._device = None

    # ------------------------------------------------------------------
    # Domain setup
    # ------------------------------------------------------------------

    def set_domain_coords(self, coords: Dict[str, np.ndarray]) -> None:
        """Set the spatial grid used for full-field inference at each tick."""
        col = [np.asarray(coords.get(c, np.zeros(1)), dtype=np.float32)
               for c in self.coord_names]
        self._domain_coords = np.column_stack(col)
        self._throttle_mask = np.ones(self._domain_coords.shape[0], dtype=bool)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def push(
        self,
        values: Dict[str, float],
        *,
        timestamp: Optional[float] = None,
        coords: Optional[Dict[str, float]] = None,
    ) -> None:
        """Push a new sensor observation into the pipeline buffer."""
        self._buffer.push({
            "values": values,
            "timestamp": timestamp or time.time(),
            "coords": coords,
        })

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _infer(self, X_np: np.ndarray) -> Dict[str, np.ndarray]:
        if self._torch is not None and hasattr(self.model, "parameters"):
            return self._torch_infer(X_np)
        out = self.model(X_np)
        if isinstance(out, np.ndarray):
            if out.ndim == 1:
                out = out[:, None]
            return {f: out[:, i] for i, f in enumerate(self.field_names) if i < out.shape[1]}
        return out

    def _torch_infer(self, X_np: np.ndarray) -> Dict[str, np.ndarray]:
        torch = self._torch
        X = torch.from_numpy(X_np).to(self._device)
        with torch.no_grad():
            if self.cfg.use_amp and self._device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = self.model(X)
            else:
                out = self.model(X)
        if hasattr(out, "y"):
            out = out.y
        if isinstance(out, dict):
            return {k: v.cpu().float().numpy() for k, v in out.items()}
        out_np = out.cpu().float().numpy()
        if out_np.ndim == 1:
            out_np = out_np[:, None]
        return {f: out_np[:, i] for i, f in enumerate(self.field_names) if i < out_np.shape[1]}

    def _get_inference_grid(self) -> Optional[np.ndarray]:
        if self._domain_coords is None:
            return None
        if self._throttle_mask is None or self._throttle_mask.all():
            return self._domain_coords
        return self._domain_coords[self._throttle_mask]

    def _apply_throttle(self) -> None:
        if self._domain_coords is None:
            return
        n = self._domain_coords.shape[0]
        current_active = int(self._throttle_mask.sum()) if self._throttle_mask is not None else n
        new_active = max(
            self.cfg.throttle_min_points,
            int(current_active * (1.0 - self.cfg.throttle_step)),
        )
        mask = np.zeros(n, dtype=bool)
        indices = np.random.choice(n, new_active, replace=False)
        mask[indices] = True
        self._throttle_mask = mask
        logger.info(
            f"RealtimePipeline: throttled grid {current_active} → {new_active} points "
            f"(utilisation={self._guard.utilisation:.2f})"
        )

    def _restore_full_grid(self) -> None:
        if self._domain_coords is not None:
            self._throttle_mask = np.ones(self._domain_coords.shape[0], dtype=bool)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Process one tick: drain buffer + run inference within budget."""
        observations = self._buffer.drain()
        grid = self._get_inference_grid()

        if grid is None and not observations:
            return

        t0 = time.time()

        state: Dict[str, Any] = {
            "timestamp": t0,
            "observations": observations,
            "fields": {},
        }

        # Full-field inference on domain grid
        if grid is not None:
            try:
                fields = self._infer(grid)
                state["fields"] = fields
                state["n_grid_points"] = grid.shape[0]
            except Exception as exc:
                logger.error(f"RealtimePipeline inference error: {exc}", exc_info=True)

        elapsed = time.time() - t0
        self._guard.record(elapsed)
        self._n_frames += 1

        # Latency policy enforcement
        if self._guard.over_budget:
            self._n_violations += 1
            policy = self.cfg.latency_policy

            if policy == "warn":
                logger.warning(
                    f"RealtimePipeline: inference {elapsed * 1000:.1f} ms > "
                    f"budget {self.cfg.data_period * 1000:.1f} ms "
                    f"(utilisation={self._guard.utilisation:.2f})"
                )
            elif policy == "skip":
                self._n_skipped += 1
                logger.debug("RealtimePipeline: skipping frame to recover latency budget.")
                return
            elif policy == "throttle":
                self._apply_throttle()
            elif policy == "raise":
                raise LatencyBudgetExceeded(
                    f"Inference {elapsed * 1000:.1f} ms > "
                    f"budget {self.cfg.data_period * 1000:.1f} ms"
                )

            if self._on_violation is not None:
                try:
                    self._on_violation(self.latency_stats())
                except Exception:
                    pass
        else:
            # Under budget — can restore full grid if throttled
            if (self.cfg.latency_policy == "throttle"
                    and self._throttle_mask is not None
                    and not self._throttle_mask.all()
                    and self._guard.utilisation < 0.7):
                self._restore_full_grid()

        # Publish state
        state["latency_ms"] = elapsed * 1000.0
        try:
            self._state_queue.put_nowait(state)
        except queue.Full:
            try:
                self._state_queue.get_nowait()
                self._state_queue.put_nowait(state)
            except queue.Empty:
                pass

        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception as exc:
                logger.warning(f"on_state callback error: {exc}")

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            self._tick()
            elapsed = time.time() - t0
            sleep = max(0.0, self.cfg.data_period - elapsed)
            time.sleep(sleep)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="realtime-pipeline"
        )
        self._thread.start()
        logger.info(
            f"RealtimePipeline started — budget={self.cfg.data_period * 1000:.1f} ms "
            f"policy='{self.cfg.latency_policy}'"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("RealtimePipeline stopped.")

    def __enter__(self) -> "RealtimePipeline":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def latest_state(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Return the most recently computed state dict, or None."""
        try:
            return self._state_queue.get(block=timeout > 0, timeout=timeout or None)
        except queue.Empty:
            return None

    def latency_stats(self) -> LatencyStats:
        s = self._guard.summary()
        return LatencyStats(
            n_frames=self._n_frames,
            n_skipped=self._n_skipped,
            n_budget_violations=self._n_violations,
            mean_inference_ms=s["mean_inference_ms"],
            max_inference_ms=s["max_inference_ms"],
            budget_ms=s["budget_ms"],
            utilisation=s["utilisation"],
        )
