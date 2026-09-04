"""SQLite-backed conversation log for every ``pinneapple_llm`` LLM call.

Zero extra dependency (``sqlite3`` is Python stdlib). Every call through
``draft.py``'s ``_call_anthropic``/``_call_openai``/``local_llm``'s
``_call_ollama`` is logged here when a store is configured -- this is
useful for two independent reasons: (1) auditability (every physics-
relevant LLM interaction this library made is inspectable and replayable,
not just trusted after the fact -- the same transparency principle
``PhysicsGuardrail`` applies to *results*, applied here to *inputs*), and
(2) it is the dataset :mod:`finetune` fine-tunes a local model on.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_DEFAULT_DB_PATH = os.path.expanduser("~/.pinneapple/conversations.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    module TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    system_prompt TEXT,
    user_prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    metadata_json TEXT
);
"""


@dataclass
class ConversationRecord:
    id: int
    timestamp: float
    module: str
    provider: str
    model: Optional[str]
    system_prompt: Optional[str]
    user_prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationStore:
    """Thin SQLite wrapper. Safe to construct repeatedly (opens/creates
    the DB file each time); not held open across calls, so it's safe to
    share a path across processes/threads without extra locking beyond
    what SQLite itself provides."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def log(
        self,
        *,
        module: str,
        provider: str,
        user_prompt: str,
        response: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "INSERT INTO conversations (timestamp, module, provider, model, system_prompt, "
                "user_prompt, response, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), module, provider, model, system_prompt, user_prompt, response,
                 json.dumps(metadata or {})),
            )
            conn.commit()
            return cur.lastrowid

    def query(self, *, module: Optional[str] = None, limit: int = 100) -> List[ConversationRecord]:
        sql = "SELECT id, timestamp, module, provider, model, system_prompt, user_prompt, response, metadata_json FROM conversations"
        params: tuple = ()
        if module is not None:
            sql += " WHERE module = ?"
            params = (module,)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params = params + (limit,)
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ConversationRecord(
                id=r[0], timestamp=r[1], module=r[2], provider=r[3], model=r[4],
                system_prompt=r[5], user_prompt=r[6], response=r[7], metadata=json.loads(r[8] or "{}"),
            )
            for r in rows
        ]

    def export_jsonl(self, out_path: str, *, module: Optional[str] = None) -> str:
        """Export every logged (prompt, response) pair as a JSONL file in
        a standard instruction-tuning shape (``{"prompt": ..., "response":
        ...}`` per line, ``system_prompt`` folded into ``prompt`` when
        present) -- ready for :mod:`finetune`."""
        records = self.query(module=module, limit=10_000_000)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for r in records:
                prompt = r.user_prompt if not r.system_prompt else f"{r.system_prompt}\n\n{r.user_prompt}"
                f.write(json.dumps({"prompt": prompt, "response": r.response}) + "\n")
        return out_path

    def __len__(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
