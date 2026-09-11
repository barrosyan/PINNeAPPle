"""Real, live integration tests for OPCUAStream/ModbusStream against
REAL local simulator servers -- not mocks. Fixes the SCADA/OPC-UA/Modbus
gap documented in the Veriphysics repo's ROADMAP.md item 4.

Each simulator (tests/simulators/opcua_simulator.py,
tests/simulators/modbus_simulator.py) is started automatically by this
file's own fixtures as a real subprocess -- no manual docker-compose
step needed, unlike the MQTT/Kafka live tests (those need real external
broker binaries; these two simulators are small, dependency-light pure-
Python servers this repo already ships). If a simulator is already
running on the expected port (started manually, e.g. for interactive
debugging), the fixture reuses it instead of starting a second instance.

Skipped (not failed) if the simulator can't be started within a
reasonable timeout, matching this codebase's live-dependency-test
convention.
"""
from __future__ import annotations

import queue
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pinneapple_systems.digital_twin.io.stream import OPCUAStream, ModbusStream

_SIMULATORS_DIR = Path(__file__).parent / "simulators"
_OPCUA_PORT = 4855
_MODBUS_PORT = 5502


def _port_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_reachable(host, port):
            return True
        time.sleep(0.2)
    return False


def _managed_simulator(script_name: str, port: int, extra_args: list):
    """A fixture body shared by both simulators: reuse an already-running
    instance on the expected port, or start + tear down a real one."""
    if _port_reachable("127.0.0.1", port):
        yield True  # already running -- don't manage its lifecycle
        return

    script = _SIMULATORS_DIR / script_name
    proc = subprocess.Popen(
        [sys.executable, str(script), "--port", str(port), *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not _wait_for_port("127.0.0.1", port, timeout=15.0):
            proc.terminate()
            pytest.skip(f"could not start {script_name} on port {port} within 15s")
        yield True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def opcua_simulator():
    yield from _managed_simulator("opcua_simulator.py", _OPCUA_PORT, [])


@pytest.fixture(scope="module")
def modbus_simulator():
    yield from _managed_simulator("modbus_simulator.py", _MODBUS_PORT, [])


def test_opcua_stream_reads_real_values_from_a_real_server(opcua_simulator):
    q: queue.Queue = queue.Queue()
    stream = OPCUAStream(
        f"opc.tcp://127.0.0.1:{_OPCUA_PORT}/pinneapple/simulator/",
        {"temperature": "ns=2;i=2", "pressure": "ns=2;i=3"},
        "sim_sensor", ["temperature", "pressure"], poll_interval=0.3,
    )
    stream.start(q)
    try:
        obs = q.get(timeout=10.0)
    finally:
        stream.stop()

    assert obs.sensor_id == "sim_sensor"
    # Real physical ranges the simulator's own sine/ramp formulas produce
    # (temperature = 20 +/- 3, pressure >= 101.3) -- not exact values,
    # since the simulator is genuinely time-varying.
    assert 15.0 <= obs.values["temperature"] <= 25.0
    assert obs.values["pressure"] >= 101.3


def test_opcua_stream_values_actually_change_over_time(opcua_simulator):
    """A real liveness check: two reads several seconds apart must not
    be bit-for-bit identical (the simulator's sine/ramp signals are
    genuinely time-varying) -- catches a stream that's silently frozen
    on its first read."""
    q: queue.Queue = queue.Queue()
    stream = OPCUAStream(
        f"opc.tcp://127.0.0.1:{_OPCUA_PORT}/pinneapple/simulator/",
        {"temperature": "ns=2;i=2", "pressure": "ns=2;i=3"},
        "sim_sensor", ["temperature", "pressure"], poll_interval=0.3,
    )
    stream.start(q)
    try:
        first = q.get(timeout=10.0)
        time.sleep(3.0)
        latest = first
        while not q.empty():
            latest = q.get_nowait()
    finally:
        stream.stop()

    assert latest.values["pressure"] > first.values["pressure"]  # monotonic ramp


def test_modbus_stream_reads_real_registers_from_a_real_server(modbus_simulator):
    q: queue.Queue = queue.Queue()
    stream = ModbusStream(
        "127.0.0.1", _MODBUS_PORT,
        {"temperature": (0, 0.01), "pressure": (1, 0.1)},
        "sim_sensor", ["temperature", "pressure"], poll_interval=0.3,
    )
    stream.start(q)
    try:
        obs = q.get(timeout=10.0)
    finally:
        stream.stop()

    assert obs.sensor_id == "sim_sensor"
    assert obs.values["temperature"] == pytest.approx(20.0, abs=0.01)
    assert obs.values["pressure"] == pytest.approx(101.3, abs=0.01)


def test_modbus_stream_respects_field_names_filter(modbus_simulator):
    """Only field_names actually listed should be read/emitted, even if
    register_map has more entries -- a real behavioral contract, not
    just a shape check."""
    q: queue.Queue = queue.Queue()
    stream = ModbusStream(
        "127.0.0.1", _MODBUS_PORT,
        {"temperature": (0, 0.01), "pressure": (1, 0.1)},
        "sim_sensor", ["temperature"], poll_interval=0.3,  # pressure excluded
    )
    stream.start(q)
    try:
        obs = q.get(timeout=10.0)
    finally:
        stream.stop()

    assert "temperature" in obs.values
    assert "pressure" not in obs.values
