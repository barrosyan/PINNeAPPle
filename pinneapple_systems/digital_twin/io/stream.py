"""Real-time data stream adapters for digital twins.

Provides adapters for multiple data sources:
- ``FileWatchStream``: polls a JSON/CSV/Parquet file for new rows
- ``MQTTStream``: subscribes to an MQTT broker (requires paho-mqtt)
- ``HTTPPollStream``: periodically polls a REST endpoint
- ``KafkaStream``: reads from Apache Kafka (requires kafka-python)
- ``OPCUAStream``: polls an OPC-UA server's tags (requires asyncua)
- ``ModbusStream``: polls a Modbus TCP server's holding registers (requires pymodbus)
- ``MockStream``: synthetic stream for testing/simulation

All streams emit ``Observation`` objects to a shared queue consumed by
the ``DigitalTwin.update_loop``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..state import Observation

logger = logging.getLogger(__name__)


class BaseStream(ABC):
    """Abstract base class for all data stream adapters."""

    def __init__(self, sensor_id: str, field_names: List[str]) -> None:
        self.sensor_id = sensor_id
        self.field_names = field_names
        self._q: Optional[queue.Queue] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def attach_queue(self, q: queue.Queue) -> None:
        self._q = q

    def _emit(self, obs: Observation) -> None:
        if self._q is not None:
            self._q.put(obs, block=False)

    def start(self, q: Optional[queue.Queue] = None) -> None:
        if q is not None:
            self._q = q
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @abstractmethod
    def _run(self) -> None: ...


# ---------------------------------------------------------------------------
# File watch stream
# ---------------------------------------------------------------------------

class FileWatchStream(BaseStream):
    """
    Polls a JSON-lines file (or CSV/Parquet) for new rows.

    Each row must have fields matching ``field_names`` and optionally
    "timestamp", "x", "y", "z" columns for spatial context.

    Parameters
    ----------
    path : str | Path      path to the data file
    poll_interval : float  seconds between file checks
    format : str           "jsonl" | "csv" | "parquet"
    """

    def __init__(
        self,
        path: str,
        sensor_id: str,
        field_names: List[str],
        *,
        poll_interval: float = 1.0,
        format: str = "jsonl",
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.path = Path(path)
        self.poll_interval = float(poll_interval)
        self.format = format
        self._last_pos: int = 0

    def _run(self) -> None:
        while self._running:
            try:
                if self.path.exists():
                    self._poll()
            except Exception as exc:
                logger.warning(f"FileWatchStream error: {exc}")
            time.sleep(self.poll_interval)

    def _poll(self) -> None:
        if self.format == "jsonl":
            self._poll_jsonl()
        elif self.format == "csv":
            self._poll_csv()
        elif self.format == "parquet":
            self._poll_parquet()

    def _poll_jsonl(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            fh.seek(self._last_pos)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._emit_row(row)
                except json.JSONDecodeError:
                    pass
            self._last_pos = fh.tell()

    def _poll_csv(self) -> None:
        try:
            import pandas as pd
            df = pd.read_csv(self.path)
            new_rows = df.iloc[self._last_pos:]
            for _, row in new_rows.iterrows():
                self._emit_row(row.to_dict())
            self._last_pos = len(df)
        except ImportError:
            logger.warning("pandas required for CSV stream. pip install pandas")

    def _poll_parquet(self) -> None:
        try:
            import pandas as pd
            df = pd.read_parquet(self.path)
            new_rows = df.iloc[self._last_pos:]
            for _, row in new_rows.iterrows():
                self._emit_row(row.to_dict())
            self._last_pos = len(df)
        except ImportError:
            logger.warning("pandas/pyarrow required for parquet stream.")

    def _emit_row(self, row: Dict[str, Any]) -> None:
        ts = float(row.get("timestamp", time.time()))
        coords: Dict[str, float] = {}
        for c in ("x", "y", "z", "t"):
            if c in row:
                coords[c] = float(row[c])
        values = {f: float(row[f]) for f in self.field_names if f in row}
        if values:
            self._emit(
                Observation(
                    timestamp=ts,
                    sensor_id=self.sensor_id,
                    values=values,
                    coords=coords or None,
                )
            )


# ---------------------------------------------------------------------------
# MQTT stream
# ---------------------------------------------------------------------------

class MQTTStream(BaseStream):
    """
    Subscribes to an MQTT topic and emits Observations.

    Requires ``paho-mqtt``: pip install paho-mqtt

    Message payload must be JSON with keys matching ``field_names``.
    """

    def __init__(
        self,
        broker: str,
        topic: str,
        sensor_id: str,
        field_names: List[str],
        *,
        port: int = 1883,
        keepalive: int = 60,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.broker = broker
        self.topic = topic
        self.port = int(port)
        self.keepalive = int(keepalive)
        self.username = username
        self.password = password
        self._client: Any = None

    def _run(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.error("paho-mqtt not installed. pip install paho-mqtt")
            return

        def on_message(_client: Any, _userdata: Any, msg: Any) -> None:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                ts = float(payload.get("timestamp", time.time()))
                values = {f: float(payload[f]) for f in self.field_names if f in payload}
                coords = {c: float(payload[c]) for c in ("x","y","z","t") if c in payload}
                if values:
                    self._emit(
                        Observation(
                            timestamp=ts,
                            sensor_id=self.sensor_id,
                            values=values,
                            coords=coords or None,
                        )
                    )
            except Exception as exc:
                logger.warning(f"MQTTStream parse error: {exc}")

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self.username:
            self._client.username_pw_set(self.username, self.password)
        self._client.on_message = on_message
        self._client.connect(self.broker, self.port, self.keepalive)
        self._client.subscribe(self.topic)
        while self._running:
            self._client.loop(timeout=1.0)
        self._client.disconnect()

    def stop(self) -> None:
        self._running = False
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        super().stop()


# ---------------------------------------------------------------------------
# HTTP polling stream
# ---------------------------------------------------------------------------

class HTTPPollStream(BaseStream):
    """
    Periodically polls a REST endpoint (GET) and emits Observations.

    The response must be JSON with keys matching ``field_names``.
    """

    def __init__(
        self,
        url: str,
        sensor_id: str,
        field_names: List[str],
        *,
        poll_interval: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
        transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.url = url
        self.poll_interval = float(poll_interval)
        self.headers = headers or {}
        self.transform = transform

    def _run(self) -> None:
        try:
            import urllib.request
        except ImportError:
            return

        while self._running:
            try:
                req = urllib.request.Request(self.url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload: Dict[str, Any] = json.loads(resp.read().decode())
                if self.transform is not None:
                    payload = self.transform(payload)
                ts = float(payload.get("timestamp", time.time()))
                values = {f: float(payload[f]) for f in self.field_names if f in payload}
                coords = {c: float(payload[c]) for c in ("x","y","z","t") if c in payload}
                if values:
                    self._emit(
                        Observation(
                            timestamp=ts,
                            sensor_id=self.sensor_id,
                            values=values,
                            coords=coords or None,
                        )
                    )
            except Exception as exc:
                logger.warning(f"HTTPPollStream error: {exc}")
            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Kafka stream
# ---------------------------------------------------------------------------

class KafkaStream(BaseStream):
    """
    Reads messages from an Apache Kafka topic.

    Requires ``kafka-python``: pip install kafka-python

    Messages must be JSON-encoded with keys matching ``field_names``.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        sensor_id: str,
        field_names: List[str],
        *,
        group_id: str = "pinneapple_dt",
        auto_offset_reset: str = "latest",
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset

    def _run(self) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError:
            logger.error("kafka-python not installed. pip install kafka-python")
            return

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        for msg in consumer:
            if not self._running:
                break
            try:
                payload = msg.value
                ts = float(payload.get("timestamp", time.time()))
                values = {f: float(payload[f]) for f in self.field_names if f in payload}
                coords = {c: float(payload[c]) for c in ("x","y","z","t") if c in payload}
                if values:
                    self._emit(
                        Observation(
                            timestamp=ts,
                            sensor_id=self.sensor_id,
                            values=values,
                            coords=coords or None,
                        )
                    )
            except Exception as exc:
                logger.warning(f"KafkaStream parse error: {exc}")
        consumer.close()


# ---------------------------------------------------------------------------
# OPC-UA stream
# ---------------------------------------------------------------------------

class OPCUAStream(BaseStream):
    """
    Polls an OPC-UA server for real-time tag values.

    Requires ``asyncua``: pip install asyncua

    OPC-UA is a pull (polled) protocol at the level this class uses it
    (reading each node's current value on an interval) -- unlike
    MQTT/Kafka, there is no server-push subscription used here, so
    ``poll_interval`` directly controls how fresh the emitted
    ``Observation``s are. (A real OPC-UA subscription -- server pushes
    on value change -- is possible via ``asyncua``'s subscription API
    but adds real complexity, e.g. a running event loop and a
    datachange callback, for a benefit -- lower latency -- this
    polling-based digital twin update loop does not need, since
    ``DigitalTwin`` itself already runs its own fixed-interval update
    loop on top of this.)

    Parameters
    ----------
    server_url : e.g. ``"opc.tcp://127.0.0.1:4840/freeopcua/server/"``
    node_ids : maps ``field_name -> a real OPC-UA NodeId string``
        (e.g. ``"ns=2;s=Temperature"`` or ``"ns=2;i=2"``) -- the caller
        must know the real node ids on the target server (browse the
        server's address space to find them; this class does not guess
        or browse for a matching node by name).
    """

    def __init__(
        self,
        server_url: str,
        node_ids: Dict[str, str],
        sensor_id: str,
        field_names: List[str],
        *,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.server_url = server_url
        self.node_ids = dict(node_ids)
        self.poll_interval = float(poll_interval)

    def _run(self) -> None:
        try:
            from asyncua.sync import Client
        except ImportError:
            logger.error("asyncua not installed. pip install asyncua")
            return

        try:
            client = Client(url=self.server_url)
            client.connect()
        except Exception as exc:
            logger.error(f"OPCUAStream: failed to connect to {self.server_url}: {exc}")
            return

        try:
            nodes = {f: client.get_node(nid) for f, nid in self.node_ids.items() if f in self.field_names}
            while self._running:
                try:
                    values = {f: float(node.read_value()) for f, node in nodes.items()}
                    if values:
                        self._emit(
                            Observation(timestamp=time.time(), sensor_id=self.sensor_id, values=values, coords=None)
                        )
                except Exception as exc:
                    logger.warning(f"OPCUAStream read error: {exc}")
                time.sleep(self.poll_interval)
        finally:
            try:
                client.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Modbus stream
# ---------------------------------------------------------------------------

class ModbusStream(BaseStream):
    """
    Polls a Modbus TCP server's holding registers for real-time values.

    Requires ``pymodbus``: pip install pymodbus

    Like :class:`OPCUAStream`, this is polling-based -- Modbus itself is
    a request/response protocol with no native push/subscribe mechanism,
    so polling is the only option, not a design shortcut taken here.

    Parameters
    ----------
    host, port : the Modbus TCP server's address (standard Modbus TCP
        port is 502; many real devices and simulators instead expose a
        non-privileged port such as 5020/5502 in test/dev setups).
    register_map : maps ``field_name -> (register_address, scale)`` --
        ``scale`` converts the raw 16-bit unsigned integer register
        value to a real physical float (e.g. a register storing
        millidegrees needs ``scale=0.001``). Every field is read as ONE
        holding register (count=1) -- multi-register (32-bit/float)
        values are not supported by this class.
    device_id : the Modbus unit/device id (also called "slave id" in
        older Modbus terminology) -- 1 by default, matching the most
        common single-device setup.
    """

    def __init__(
        self,
        host: str,
        port: int,
        register_map: Dict[str, "tuple[int, float]"],
        sensor_id: str,
        field_names: List[str],
        *,
        poll_interval: float = 1.0,
        device_id: int = 1,
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.host = host
        self.port = int(port)
        self.register_map = dict(register_map)
        self.poll_interval = float(poll_interval)
        self.device_id = int(device_id)

    def _run(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            logger.error("pymodbus not installed. pip install pymodbus")
            return

        client = ModbusTcpClient(self.host, port=self.port)
        if not client.connect():
            logger.error(f"ModbusStream: failed to connect to {self.host}:{self.port}")
            return

        try:
            while self._running:
                try:
                    values: Dict[str, float] = {}
                    for f, (address, scale) in self.register_map.items():
                        if f not in self.field_names:
                            continue
                        result = client.read_holding_registers(address, count=1, device_id=self.device_id)
                        if result.isError():
                            logger.warning(f"ModbusStream: error reading register {address} for field {f!r}")
                            continue
                        values[f] = float(result.registers[0]) * scale
                    if values:
                        self._emit(
                            Observation(timestamp=time.time(), sensor_id=self.sensor_id, values=values, coords=None)
                        )
                except Exception as exc:
                    logger.warning(f"ModbusStream read error: {exc}")
                time.sleep(self.poll_interval)
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Mock stream (for testing / simulation)
# ---------------------------------------------------------------------------

class MockStream(BaseStream):
    """
    Synthetic data stream for testing.

    A ``generator_fn(t)`` is called at each tick and should return
    ``{field_name: value}`` for the simulated sensor.
    """

    def __init__(
        self,
        sensor_id: str,
        field_names: List[str],
        generator_fn: Callable[[float], Dict[str, float]],
        *,
        tick_interval: float = 0.1,
        coords: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(sensor_id, field_names)
        self.generator_fn = generator_fn
        self.tick_interval = float(tick_interval)
        self.coords = coords

    def _run(self) -> None:
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            try:
                values = self.generator_fn(t)
                self._emit(
                    Observation(
                        timestamp=time.time(),
                        sensor_id=self.sensor_id,
                        values=values,
                        coords=self.coords,
                    )
                )
            except Exception as exc:
                logger.warning(f"MockStream generator error: {exc}")
            time.sleep(self.tick_interval)
