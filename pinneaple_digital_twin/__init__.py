"""pinneaple_digital_twin — Digital twins built on top of pinneaple surrogate models.

Key capabilities
----------------
- Wrap any trained model (PINN, GNN, DeepONet, FNO, …) as a digital twin.
- Ingest real-time sensor data from MQTT, Kafka, REST APIs, files, or mocks.
- Fuse predictions with observations using Kalman-family filters (EKF, EnKF).
- Detect anomalies via threshold, z-score, and Mahalanobis distance checks.
- Maintain rolling history of state estimates.
- Callback hooks for custom monitoring / dashboards.

Quick start
-----------
>>> from pinneaple_digital_twin import DigitalTwin, build_digital_twin
>>> from pinneaple_digital_twin.io import MockStream
>>> import numpy as np

>>> model = ...  # your trained surrogate model
>>> dt = build_digital_twin(model, field_names=["u", "v", "p"])
>>> stream = MockStream(
...     "inlet", ["u", "v"],
...     lambda t: {"u": 1.0 + 0.1 * np.sin(t), "v": 0.0},
...     coords={"x": 0.0, "y": 0.5},
... )
>>> dt.add_stream(stream)
>>> x = np.linspace(0, 1, 200)
>>> y = np.linspace(0, 1, 200)
>>> xx, yy = np.meshgrid(x, y)
>>> dt.set_domain_coords({"x": xx.ravel(), "y": yy.ravel()})
>>> with dt:
...     import time; time.sleep(5)   # twin runs for 5 s
... print(dt.state.fields["u"].mean())
"""

from .twin import DigitalTwin, DigitalTwinConfig, build_digital_twin
from .state import SystemState, Observation
from .io import (
    Sensor, SensorRegistry,
    BaseStream, FileWatchStream, MQTTStream,
    HTTPPollStream, KafkaStream, MockStream,
)
from .assimilation import ExtendedKalmanFilter, EnsembleKalmanFilter
from .monitoring import (
    AnomalyEvent, AnomalyMonitor,
    ThresholdDetector, ZScoreDetector, MahalanobisDetector,
)

__all__ = [
    # Core
    "DigitalTwin",
    "DigitalTwinConfig",
    "build_digital_twin",
    # State
    "SystemState",
    "Observation",
    # I/O
    "Sensor",
    "SensorRegistry",
    "BaseStream",
    "FileWatchStream",
    "MQTTStream",
    "HTTPPollStream",
    "KafkaStream",
    "MockStream",
    # Assimilation
    "ExtendedKalmanFilter",
    "EnsembleKalmanFilter",
    # Monitoring
    "AnomalyEvent",
    "AnomalyMonitor",
    "ThresholdDetector",
    "ZScoreDetector",
    "MahalanobisDetector",
]
