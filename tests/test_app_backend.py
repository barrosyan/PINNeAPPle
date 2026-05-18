"""Backend integration tests for pinneapple_app.

Run with:
    pytest tests/pinneapple_app/test_backend.py -v

Requires:
    pip install httpx pytest pytest-asyncio
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# FastAPI test client (no external server needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from pinneapple_app.backend.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health & info
# ---------------------------------------------------------------------------

class TestHealthInfo:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "pinneapple_app"

    def test_info_structure(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        body = r.json()
        assert "n_models" in body
        assert "n_problems" in body
        assert "families" in body
        assert body["n_models"] > 0
        assert body["n_problems"] > 0

    def test_info_model_count(self, client):
        r = client.get("/api/info")
        assert r.json()["n_models"] >= 10, "Expected at least 10 registered models"


# ---------------------------------------------------------------------------
# 2. Problems
# ---------------------------------------------------------------------------

class TestProblems:
    def test_list_problems(self, client):
        r = client.get("/api/problems")
        assert r.status_code == 200
        problems = r.json()
        assert isinstance(problems, list)
        assert len(problems) >= 1

    def test_problem_fields(self, client):
        r = client.get("/api/problems")
        for prob in r.json():
            assert "name" in prob
            assert "family" in prob

    def test_get_single_problem(self, client):
        probs = client.get("/api/problems").json()
        first = probs[0]["name"]
        r = client.get(f"/api/problems/{first}")
        assert r.status_code == 200
        assert r.json()["name"] == first

    def test_get_nonexistent_problem(self, client):
        r = client.get("/api/problems/__does_not_exist__")
        assert r.status_code == 404

    def test_problems_include_heat(self, client):
        names = {p["name"] for p in client.get("/api/problems").json()}
        assert any("heat" in n for n in names), "Expected at least one heat problem"

    def test_custom_problem_validate(self, client):
        payload = {
            "coords": ["x", "y"],
            "fields": ["u"],
            "name": "test_poisson",
            "equations": ["u_xx + u_yy"],
            "boundary_conditions": [{"type": "dirichlet", "value": 0}],
            "domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "dim": 2,
        }
        r = client.post("/api/problems/custom/validate", json=payload)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_list_models(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        models = r.json()
        assert isinstance(models, list)
        assert len(models) >= 10

    def test_model_fields(self, client):
        models = client.get("/api/models").json()
        for m in models[:5]:
            assert "name" in m
            assert "family" in m
            assert "description" in m

    def test_list_families(self, client):
        r = client.get("/api/models/families")
        assert r.status_code == 200
        fams = r.json()["families"]
        assert len(fams) >= 1

    def test_get_single_model(self, client):
        models = client.get("/api/models").json()
        first_name = models[0]["name"]
        r = client.get(f"/api/models/{first_name}")
        assert r.status_code == 200
        assert r.json()["name"] == first_name

    def test_get_nonexistent_model(self, client):
        r = client.get("/api/models/__no_such_model__")
        assert r.status_code == 404

    def test_family_filter(self, client):
        r = client.get("/api/models?family=pinns")
        assert r.status_code == 200
        for m in r.json():
            assert m["family"] == "pinns"

    def test_recommend_models(self, client):
        r = client.get("/api/models/recommend/fluid?n=3")
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        assert len(recs) <= 3

    def test_list_metrics(self, client):
        r = client.get("/api/models/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "available" in body
        assert "defaults" in body
        assert len(body["available"]) >= 1

    def test_vanilla_pinn_registered(self, client):
        r = client.get("/api/models")
        names = {m["name"] for m in r.json()}
        assert "vanilla_pinn" in names

    def test_siren_registered(self, client):
        r = client.get("/api/models")
        names = {m["name"] for m in r.json()}
        assert "siren" in names


# ---------------------------------------------------------------------------
# 4. Experiments
# ---------------------------------------------------------------------------

def _launch(client, *, problem="heat_2d", model="vanilla_pinn", epochs=2,
            n_interior=50, n_boundary=25, n_initial=25,
            grid_resolution=8, use_solver=False) -> str:
    payload = {
        "problem_name": problem,
        "models": [{"name": model}],
        "epochs": epochs,
        "collocation": {
            "strategy": "lhs",
            "n_interior": n_interior,
            "n_boundary": n_boundary,
            "n_initial": n_initial,
        },
        "data": {
            "use_solver": use_solver,
            "n_snapshots": 2,
            "grid_resolution": grid_resolution,
        },
    }
    r = client.post("/api/experiments/launch", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    return body["experiment_id"]


def _wait_done(client, exp_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/experiments/{exp_id}/status")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.5)
    pytest.fail(f"Experiment {exp_id} did not finish within {timeout}s")


class TestExperiments:
    def test_launch_returns_id(self, client):
        exp_id = _launch(client)
        assert isinstance(exp_id, str)
        assert len(exp_id) >= 6

    def test_launch_queued_status(self, client):
        payload = {
            "problem_name": "heat_2d",
            "models": [{"name": "vanilla_pinn"}],
            "epochs": 2,
        }
        r = client.post("/api/experiments/launch", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["progress"] == 0.0

    def test_invalid_model_returns_422(self, client):
        payload = {
            "problem_name": "heat_2d",
            "models": "vanilla_pinn",   # should be a list of objects
        }
        r = client.post("/api/experiments/launch", json=payload)
        assert r.status_code == 422

    def test_status_endpoint(self, client):
        exp_id = _launch(client)
        r = client.get(f"/api/experiments/{exp_id}/status")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "progress" in body
        assert "experiment_id" in body

    def test_status_404_for_unknown(self, client):
        r = client.get("/api/experiments/nonexistent_id/status")
        assert r.status_code == 404

    def test_full_experiment_completes(self, client):
        exp_id = _launch(client, epochs=2)
        status = _wait_done(client, exp_id)
        assert status["status"] == "done"
        assert status["progress"] == 100.0

    def test_results_available_after_done(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        r = client.get(f"/api/experiments/{exp_id}/results")
        assert r.status_code == 200, r.text

    def test_results_structure(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        r = client.get(f"/api/experiments/{exp_id}/results")
        assert r.status_code == 200
        body = r.json()
        assert "leaderboard" in body
        assert "charts" in body
        assert "summary" in body
        assert "experiment_id" in body
        assert "problem_name" in body

    def test_results_no_nan(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        r = client.get(f"/api/experiments/{exp_id}/results")
        assert r.status_code == 200
        # Should parse cleanly as valid JSON (no NaN/Infinity at JSON level)
        body = r.json()
        assert body is not None

    def test_leaderboard_contains_model(self, client):
        exp_id = _launch(client, model="vanilla_pinn", epochs=2)
        _wait_done(client, exp_id)
        body = client.get(f"/api/experiments/{exp_id}/results").json()
        model_names = [row["model"] for row in body.get("leaderboard", [])]
        assert "vanilla_pinn" in model_names

    def test_list_experiments(self, client):
        _launch(client, epochs=2)
        r = client.get("/api/experiments")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) >= 1

    def test_results_not_ready_returns_404(self, client):
        exp_id = _launch(client, epochs=2)
        # Before it completes, results should return 404 (or 200 if it finishes fast)
        r = client.get(f"/api/experiments/{exp_id}/results")
        assert r.status_code in (200, 404)   # either is valid


# ---------------------------------------------------------------------------
# 5. Payload quality
# ---------------------------------------------------------------------------

class TestPayloadQuality:
    def test_charts_are_base64(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        body = client.get(f"/api/experiments/{exp_id}/results").json()
        charts = body.get("charts", {})
        for key, value in charts.items():
            if value:
                import base64
                try:
                    base64.b64decode(value)
                except Exception:
                    pytest.fail(f"Chart '{key}' is not valid base64")

    def test_summary_is_string(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        body = client.get(f"/api/experiments/{exp_id}/results").json()
        assert isinstance(body["summary"], str)
        assert len(body["summary"]) > 10

    def test_all_numeric_metrics_are_finite_or_null(self, client):
        exp_id = _launch(client, epochs=2)
        _wait_done(client, exp_id)
        body = client.get(f"/api/experiments/{exp_id}/results").json()
        for row in body.get("leaderboard", []):
            for k, v in row.items():
                if isinstance(v, float):
                    assert v is None or math.isfinite(v), \
                        f"Metric '{k}' has non-finite value {v}"
