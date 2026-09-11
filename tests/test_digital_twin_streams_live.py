"""Real, live integration tests for MQTTStream/KafkaStream against REAL
local brokers -- not mocks, not MockStream. Fixes the gap documented in
the Veriphysics repo's ROADMAP.md item 6.

Requires the brokers from tests/docker/docker-compose.mqtt-kafka.yml:

    docker compose -f tests/docker/docker-compose.mqtt-kafka.yml up -d
    .venv/bin/python3 -m pytest tests/test_digital_twin_streams_live.py -q
    docker compose -f tests/docker/docker-compose.mqtt-kafka.yml down -v

Skipped (not failed) if a broker isn't reachable, matching this
codebase's convention for every other live-dependency test (Ollama,
etc).
"""
from __future__ import annotations

import json
import queue
import socket
import time

import pytest

from pinneapple_systems.digital_twin.io.stream import MQTTStream, KafkaStream

_MQTT_HOST, _MQTT_PORT = "127.0.0.1", 1883
_KAFKA_BOOTSTRAP = "127.0.0.1:9092"


def _port_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _mqtt_reachable() -> bool:
    return _port_reachable(_MQTT_HOST, _MQTT_PORT)


def _kafka_reachable() -> bool:
    return _port_reachable("127.0.0.1", 9092)


_skip_no_mqtt = pytest.mark.skipif(
    not _mqtt_reachable(),
    reason="no local MQTT broker on 127.0.0.1:1883 -- "
           "docker compose -f tests/docker/docker-compose.mqtt-kafka.yml up -d",
)
_skip_no_kafka = pytest.mark.skipif(
    not _kafka_reachable(),
    reason="no local Kafka-API broker on 127.0.0.1:9092 -- "
           "docker compose -f tests/docker/docker-compose.mqtt-kafka.yml up -d",
)


@_skip_no_mqtt
def test_mqtt_stream_receives_a_real_published_message():
    import paho.mqtt.client as mqtt

    topic = "pinneapple/test/sensor1"
    q: queue.Queue = queue.Queue()
    stream = MQTTStream(_MQTT_HOST, topic, "sensor1", ["u", "v"], port=_MQTT_PORT)
    stream.start(q)
    time.sleep(1.0)  # let the subscriber thread actually connect + subscribe

    try:
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        publisher.connect(_MQTT_HOST, _MQTT_PORT, 60)
        publisher.loop_start()
        publisher.publish(topic, json.dumps({"u": 1.5, "v": -0.3, "timestamp": time.time()}))
        time.sleep(0.5)
        publisher.loop_stop()
        publisher.disconnect()

        obs = q.get(timeout=10.0)
    finally:
        stream.stop()

    assert obs.sensor_id == "sensor1"
    assert obs.values["u"] == pytest.approx(1.5)
    assert obs.values["v"] == pytest.approx(-0.3)


@_skip_no_mqtt
def test_mqtt_stream_ignores_malformed_payload_without_crashing():
    """A real robustness check: a genuinely malformed message on the
    topic must not kill the subscriber thread -- the next valid message
    must still arrive."""
    import paho.mqtt.client as mqtt

    topic = "pinneapple/test/sensor_malformed"
    q: queue.Queue = queue.Queue()
    stream = MQTTStream(_MQTT_HOST, topic, "sensor_malformed", ["u"], port=_MQTT_PORT)
    stream.start(q)
    time.sleep(1.0)

    try:
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        publisher.connect(_MQTT_HOST, _MQTT_PORT, 60)
        publisher.loop_start()
        publisher.publish(topic, "not valid json at all")
        time.sleep(0.3)
        publisher.publish(topic, json.dumps({"u": 7.0, "timestamp": time.time()}))
        time.sleep(0.5)
        publisher.loop_stop()
        publisher.disconnect()

        obs = q.get(timeout=10.0)
    finally:
        stream.stop()

    assert obs.values["u"] == pytest.approx(7.0)


@_skip_no_kafka
def test_kafka_stream_receives_a_real_produced_message():
    from kafka import KafkaProducer

    topic = "pinneapple-test-topic-1"
    q: queue.Queue = queue.Queue()
    stream = KafkaStream(_KAFKA_BOOTSTRAP, topic, "sensor2", ["u"], auto_offset_reset="earliest")
    stream.start(q)
    time.sleep(3.0)  # real consumer-group join/partition-assignment latency

    producer = None
    try:
        producer = KafkaProducer(
            bootstrap_servers=_KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        producer.send(topic, {"u": 42.0, "timestamp": time.time()})
        producer.flush()

        obs = q.get(timeout=20.0)
    finally:
        stream.stop()
        if producer is not None:
            producer.close()

    assert obs.sensor_id == "sensor2"
    assert obs.values["u"] == pytest.approx(42.0)
