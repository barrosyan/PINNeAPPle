"""Centralised session-state helpers for the Pinneaple app."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


_DEFAULTS: Dict[str, Any] = {
    "projects":       [],     # list[ProjectDict]
    "active_project": None,   # str project id | None
    "notifications":  [],     # list[{msg, level}]
}


def init():
    """Initialise all missing session-state keys with defaults."""
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Projects ──────────────────────────────────────────────────────────────────

def new_project(name: str, problem: Optional[dict] = None) -> str:
    """Create a new project, make it active, and return its id."""
    init()
    pid = str(uuid.uuid4())[:8]
    st.session_state.projects.append({
        "id":         pid,
        "name":       name,
        "problem":    problem or {},
        "model_config":   None,
        "trained_model":  None,
        "simulation":     None,
        "collocation":    None,
        "geometry":       None,
        "timeseries_data": None,
        "timeseries_col":  None,
        "inference_result": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state.active_project = pid
    return pid


def get_project(pid: str) -> Optional[Dict]:
    init()
    for p in st.session_state.projects:
        if p["id"] == pid:
            return p
    return None


def update_project(data: dict):
    """Update the active project with the supplied dict."""
    init()
    pid = st.session_state.active_project
    if pid is None:
        return
    for p in st.session_state.projects:
        if p["id"] == pid:
            p.update(data)
            return


def active_project() -> Optional[Dict]:
    init()
    pid = st.session_state.active_project
    return get_project(pid) if pid else None


# ── Notifications ─────────────────────────────────────────────────────────────

def notify(msg: str, level: str = "info"):
    """Queue a notification banner (level: info | success | warning | error)."""
    init()
    st.session_state.notifications.append({"msg": msg, "level": level})


def pop_notifications() -> List[Dict]:
    """Consume and return all pending notifications."""
    init()
    n = list(st.session_state.notifications)
    st.session_state.notifications = []
    return n
