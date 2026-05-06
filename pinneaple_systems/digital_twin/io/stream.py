"""Real-time data stream adapters for digital twins.

Provides adapters for multiple data sources:
- ``FileWatchStream``: polls a JSON/CSV/Parquet file for new rows
- ``MQTTStream``: subscribes to an MQTT broker (requires paho-mqtt)
- ``HTTPPollStream``: periodically polls a REST endpoint
- ``KafkaStream``: reads from Apache Kafka (requires kafka-python)
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

        self._client = mqtt.Client()
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
        group_id: str = "pinneaple_dt",
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
