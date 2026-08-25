"""pinneapple_systems.component_modeling.control — PID feedback control for
an arbitrary differentiable or black-box component.

Standard control-theory utility: ``PIDController`` implements the textbook
discrete-time law u = Kp*e + Ki*integral(e) + Kd*de/dt, and
``run_closed_loop()`` closes a loop around any single-input/single-output
callable "plant" (a bare Python function, a lambda wrapping one channel of a
trained model, a toy synthetic system, ...). Deliberately decoupled from any
specific model registry or component type — the plant only needs to be a
``float -> float`` callable.

State estimation is NOT reimplemented here: pinneapple already ships
Kalman-family estimators — see ``pinneapple_systems.digital_twin.assimilation
.kalman`` (EKF/EnKF for live digital twins) and
``pinneapple_neural.architectures.classical_ts.kalman`` (time-series Kalman/
UKF/EnKF). Wrap one of those into a ``float -> float`` callable and pass it
as ``observer`` below instead of adding a third parallel implementation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


class PIDController:
    """Standard discrete-time PID: u = Kp*e + Ki*integral(e) + Kd*de/dt,
    e = setpoint - measurement."""

    def __init__(self, kp: float, ki: float, kd: float, setpoint: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


def run_closed_loop(
    plant: Callable[[float], float],
    *,
    setpoint: float,
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.0,
    steps: int = 50,
    dt: float = 0.1,
    process_noise: float = 0.0,
    measurement_noise: float = 0.0,
    observer: Optional[Callable[[float], float]] = None,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Closes a discrete-time control loop around ``plant``.

    At each step the PID controller computes an action from the (optionally
    filtered) measurement, ``plant(action)`` returns the new output, and the
    loop repeats. ``plant`` has no notion of "components" or a registry — it
    is any scalar callable.

    ``observer``, if given, must be a ``float -> float`` callable mapping a
    raw (noisy) measurement to a filtered estimate — e.g.
    ``lambda y: kf.step(np.array([y]))["x"][0]`` wrapping an
    ``ExtendedKalmanFilter``. Without one, the controller acts on the raw
    measurement directly.
    """
    rng = np.random.default_rng(seed)
    controller = PIDController(kp=kp, ki=ki, kd=kd, setpoint=setpoint)

    history: Dict[str, List[float]] = {
        "time": [], "setpoint": [], "action": [], "measured": [], "filtered": [],
    }
    filtered = 0.0
    for step in range(steps):
        action = controller.update(filtered, dt)
        true_output = float(plant(action)) + rng.normal(0, process_noise)
        measured = true_output + rng.normal(0, measurement_noise)
        filtered = float(observer(measured)) if observer is not None else measured

        history["time"].append(step * dt)
        history["setpoint"].append(setpoint)
        history["action"].append(action)
        history["measured"].append(measured)
        history["filtered"].append(filtered)

    return history
