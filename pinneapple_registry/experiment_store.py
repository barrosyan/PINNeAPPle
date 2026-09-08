"""pinneapple_registry.experiment_store — ``ExperimentStore``: a SQLite-
backed run tracker. Closes the one gap ``pinneapple_hub`` genuinely has no
answer for: it publishes finished models to the Hugging Face Hub, but
nothing in the repo durably records *runs* (start/metrics-per-epoch/finish)
independent of whatever a training script happens to print to stdout.

Deliberately minimal — a run tracker, not a full experiment-management
UI. ``ExperimentStore`` does not train anything; callers (e.g.
``ComponentModel.fit()``, an orchestration flow, or a plain script) log
into it explicitly.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id TEXT NOT NULL,
    name TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    final_metrics TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    step INTEGER,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    logged_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


class ExperimentStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self.db_path = db_path
        with closing(sqlite3.connect(db_path)) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def find_or_create_experiment(self, problem_id: str, name: Optional[str] = None) -> int:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT id FROM experiments WHERE problem_id=? AND name IS ?", (problem_id, name)
            ).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO experiments (problem_id, name, created_at) VALUES (?, ?, ?)",
                (problem_id, name, datetime.now().isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def start_run(self, problem_id: str, name: Optional[str] = None) -> int:
        experiment_id = self.find_or_create_experiment(problem_id, name)
        with closing(self._conn()) as conn:
            cur = conn.execute(
                "INSERT INTO runs (experiment_id, status, started_at) VALUES (?, 'running', ?)",
                (experiment_id, datetime.now().isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def log_metrics(self, run_id: int, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        now = datetime.now().isoformat()
        with closing(self._conn()) as conn:
            conn.executemany(
                "INSERT INTO metrics (run_id, step, name, value, logged_at) VALUES (?, ?, ?, ?, ?)",
                [(run_id, step, k, float(v), now) for k, v in metrics.items()],
            )
            conn.commit()

    def finish_run(self, run_id: int, final_metrics: Optional[Dict[str, float]] = None, status: str = "finished") -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                "UPDATE runs SET status=?, finished_at=?, final_metrics=? WHERE id=?",
                (status, datetime.now().isoformat(), json.dumps(final_metrics or {}), run_id),
            )
            conn.commit()

    def run_metrics(self, run_id: int) -> List[Dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT step, name, value, logged_at FROM metrics WHERE run_id=?", (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_best_run(self, problem_id: str, metric: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """Best run for ``problem_id`` by a metric found in ``final_metrics``.
        ``mode`` is ``"min"`` or ``"max"``."""
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """SELECT r.* FROM runs r JOIN experiments e ON r.experiment_id = e.id
                   WHERE e.problem_id = ? AND r.final_metrics IS NOT NULL""",
                (problem_id,),
            ).fetchall()

        best: Optional[Dict[str, Any]] = None
        best_val: Optional[float] = None
        for row in rows:
            metrics = json.loads(row["final_metrics"] or "{}")
            if metric not in metrics:
                continue
            val = float(metrics[metric])
            if best_val is None or (val < best_val if mode == "min" else val > best_val):
                best_val, best = val, {**dict(row), "final_metrics": metrics}
        return best

    def list_runs(self, problem_id: str) -> List[Dict[str, Any]]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """SELECT r.* FROM runs r JOIN experiments e ON r.experiment_id = e.id
                   WHERE e.problem_id = ? ORDER BY r.started_at""",
                (problem_id,),
            ).fetchall()
            out = []
            for row in rows:
                d = dict(row)
                d["final_metrics"] = json.loads(d["final_metrics"]) if d.get("final_metrics") else None
                out.append(d)
            return out
