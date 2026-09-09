"""Tests for pinneapple_app.backend.routers.admin and the /api/metrics
endpoint. Matches the TestClient pattern already used by
tests/test_app_backend.py.

Run with: pytest tests/test_admin_router.py -v
Requires: pip install httpx (same pre-existing requirement as
tests/test_app_backend.py — this environment does not have it installed,
so these tests error the same way that file's already do here; not a
regression this file introduces).
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("PINNEAPPLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    from fastapi.testclient import TestClient
    from pinneapple_app.backend.main import app
    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_is_prometheus_text_format(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    assert "pinneapple_app_uptime_seconds" in resp.text
    assert "pinneapple_app_requests_total" in resp.text


def test_admin_summary_requires_token(client):
    resp = client.get("/api/admin/summary")
    assert resp.status_code == 401


def test_admin_summary_with_valid_token(client):
    resp = client.get("/api/admin/summary", headers={"X-Admin-Token": "test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert "registry_root" in body
    assert "models" in body


def test_admin_disabled_without_env_var(client, monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_ADMIN_TOKEN", raising=False)
    resp = client.get("/api/admin/summary", headers={"X-Admin-Token": "anything"})
    assert resp.status_code == 503


def test_admin_promote_round_trip(client, tmp_path):
    from pinneapple_registry import ArtifactRegistry
    from pinneapple_systems.component_library import ComponentRegistry

    registry_root = os.environ["PINNEAPPLE_REGISTRY_ROOT"]
    registry = ArtifactRegistry(registry_root)
    model = ComponentRegistry.build("ValveModifiedMLP")
    version = registry.models.save("valve_admin_demo", model)

    resp = client.post(
        "/api/admin/models/promote",
        json={"problem_id": "valve_admin_demo", "version": version, "stage": "production"},
        headers={"X-Admin-Token": "test-token"},
    )
    assert resp.status_code == 200
    assert registry.models.metadata("valve_admin_demo", version)["stage"] == "production"
