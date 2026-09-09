"""pinneapple_registry.experiment_memory -- ``ExperimentMemory``: persistent
cross-run memory for an (separately built, not part of this repo) autonomous
research loop.

Why this exists: a research loop that starts every new run from zero re-tries
approaches that already failed, and re-discovers lessons that a previous run
already learned. This module gives such a loop a place to ask, before it
commits compute to a new attempt, "has something like this been tried
before, and what happened?" -- by logging every attempt as a durable
``ExperimentRecord`` and letting later runs query for similar past attempts
and their outcomes/lessons.

This module does not implement its own storage. ``ExperimentMemory`` is a
thin layer on top of the real ``pinneapple_registry.experiment_store
.ExperimentStore`` (a SQLite-backed run tracker): every ``ExperimentRecord``
is logged via ``ExperimentStore.start_run``/``log_metrics``/``finish_run``
and read back via ``ExperimentStore.list_runs``, exactly the same durable
on-disk store the rest of ``pinneapple_registry`` already uses. The only
extra thing this module does beyond calling that real API is (a) pack the
free-text/record fields that ``ExperimentStore`` has no dedicated columns
for into the ``final_metrics`` JSON blob it already stores per-run, and (b)
a small read-only ``SELECT DISTINCT problem_id FROM experiments`` against
the store's own schema, needed because ``ExperimentStore`` has no
"list every problem_id it has ever seen" method -- that query reads the
same tables ``ExperimentStore`` created and writes nothing.

Honesty note on similarity search: ``query_similar`` is **lexical**
similarity -- TF-IDF cosine similarity over ``problem_description`` text
(via ``sklearn.feature_extraction.text.TfidfVectorizer``, a real,
already-present dependency of this repo), with a small score boost when
``physics_domain`` also matches. This finds past experiments that share
vocabulary with the query, e.g. "airfoil AoA sweep diverges" will surface
past records mentioning "AoA" and "diverge". It is NOT semantic/embedding
search and will not, by itself, connect two descriptions that mean the same
thing in different words (e.g. "angle of attack" vs. "AoA" only match
because the strings share a token in this example -- a true paraphrase with
no shared vocabulary would not match). Callers that need that should route
the description text through a real embedding model themselves; this module
does not add one, to avoid pulling in a new heavy ML dependency for a
capability this repo does not otherwise need.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .experiment_store import ExperimentStore

__all__ = ["ExperimentRecord", "ExperimentMemory"]


@dataclass
class ExperimentRecord:
    """One logged attempt at solving a physics problem.

    Attributes
    ----------
    problem_description : free-text NL goal/description that started the
        run (e.g. "simulate lift/drag for a NACA 0012 airfoil across an AoA
        sweep from 0 to 25 degrees").
    physics_domain : a short tag/keyword for the physics area, e.g.
        "heat_conduction", "cfd", "orbital_mechanics". This module never
        infers or invents this tag -- it is whatever the caller passes.
    approach_summary : what was tried (solver/method/model used), e.g.
        "LBM D2Q9 with Smagorinsky LES, Cs=0.17".
    outcome : "success", "failure", or "partial".
    confidence_score : optional plain float. Callers that already computed
        a real score via ``pinneapple_analysis.verification
        .physics_confidence_score.PhysicsConfidenceScore`` pass its
        numeric value here; this module takes a plain ``Optional[float]``
        rather than importing that module, so logging an experiment never
        requires depending on it.
    metrics : dict of numeric metrics for the run (e.g. {"rmse": 0.02}).
    timestamp : ISO-8601 string; defaults to "now" at construction time.
    lessons_learned : optional free text, e.g. "AoA sweep above 20deg
        diverged with LBM Smagorinsky Cs=0.17, needed Cs=0.1".
    """

    problem_description: str
    physics_domain: str
    approach_summary: str
    outcome: str
    confidence_score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    lessons_learned: Optional[str] = None


def _record_to_payload(record: ExperimentRecord) -> Dict[str, Any]:
    return {
        "problem_description": record.problem_description,
        "physics_domain": record.physics_domain,
        "approach_summary": record.approach_summary,
        "outcome": record.outcome,
        "confidence_score": record.confidence_score,
        "metrics": dict(record.metrics or {}),
        "timestamp": record.timestamp,
        "lessons_learned": record.lessons_learned,
    }


def _payload_to_record(payload: Dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        problem_description=payload.get("problem_description", ""),
        physics_domain=payload.get("physics_domain", "unknown"),
        approach_summary=payload.get("approach_summary", ""),
        outcome=payload.get("outcome", ""),
        confidence_score=payload.get("confidence_score"),
        metrics=dict(payload.get("metrics") or {}),
        timestamp=payload.get("timestamp", ""),
        lessons_learned=payload.get("lessons_learned"),
    )


class ExperimentMemory:
    """Persistent memory of past experiment attempts, backed by the real
    ``ExperimentStore``.

    Construct either from an existing ``ExperimentStore`` (e.g.
    ``ArtifactRegistry.experiments``, so the research loop shares the same
    on-disk DB the rest of the registry uses)::

        memory = ExperimentMemory(store=registry.experiments)

    or standalone from a DB path::

        memory = ExperimentMemory(db_path="./my_registry/experiments.db")
    """

    def __init__(self, store: Optional[ExperimentStore] = None, *, db_path: Optional[str] = None):
        if store is not None and db_path is not None:
            raise ValueError("Pass either `store` or `db_path`, not both.")
        if store is None:
            if db_path is None:
                raise ValueError("ExperimentMemory requires either `store` or `db_path`.")
            store = ExperimentStore(db_path)
        self._store = store

    # -- logging -------------------------------------------------------

    def log_experiment(self, record: ExperimentRecord) -> str:
        """Logs ``record`` via the real ``ExperimentStore`` run-tracking
        API (``start_run`` / ``log_metrics`` / ``finish_run``) and returns
        the store's real run id (as a string) -- a genuine, durable,
        queryable identifier, not one minted by this module.
        """
        problem_id = record.physics_domain or "unknown"
        run_id = self._store.start_run(problem_id, name=record.problem_description)
        if record.metrics:
            self._store.log_metrics(run_id, record.metrics)
        payload = _record_to_payload(record)
        self._store.finish_run(run_id, final_metrics=payload, status=record.outcome)
        return str(run_id)

    # -- retrieval -------------------------------------------------------

    def _list_problem_ids(self) -> List[str]:
        """Every distinct ``problem_id`` (== physics_domain) the store has
        ever seen. ``ExperimentStore`` has no method for this, so this
        reads the store's own ``experiments`` table directly -- read-only,
        same schema, no parallel storage of our own."""
        with closing(sqlite3.connect(self._store.db_path)) as conn:
            rows = conn.execute("SELECT DISTINCT problem_id FROM experiments").fetchall()
        return [row[0] for row in rows]

    def _all_records(self) -> List[ExperimentRecord]:
        """Every ``ExperimentRecord`` logged so far, reconstructed from the
        real ``ExperimentStore.list_runs`` output. Runs with no
        ``final_metrics`` (i.e. never finished via ``log_experiment``) are
        skipped -- they carry no record payload to reconstruct."""
        records: List[ExperimentRecord] = []
        for problem_id in self._list_problem_ids():
            for run in self._store.list_runs(problem_id):
                payload = run.get("final_metrics")
                if not payload:
                    continue
                records.append(_payload_to_record(payload))
        return records

    def query_similar(
        self,
        problem_description: str,
        *,
        physics_domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[ExperimentRecord, float]]:
        """Real, honest **lexical** similarity search: TF-IDF cosine
        similarity of ``problem_description`` against every stored
        record's ``problem_description``, with a small additive score
        boost when ``physics_domain`` also matches (exact match: +0.25,
        substring match either direction: +0.1, capped at 1.0). This is
        vocabulary overlap, not semantic understanding -- see the module
        docstring.

        Returns up to ``top_k`` ``(record, score)`` pairs, highest score
        first. Returns ``[]`` if nothing has been logged yet, or if the
        query/corpus share no vocabulary the vectorizer can use.
        """
        records = self._all_records()
        if not records:
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [r.problem_description for r in records]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(corpus + [problem_description])
        except ValueError:
            # Empty vocabulary after stop-word removal (e.g. query and
            # corpus share no meaningful tokens) -- honestly no lexical
            # signal to rank on.
            return []

        query_vec = tfidf[-1]
        corpus_vecs = tfidf[:-1]
        sims = cosine_similarity(query_vec, corpus_vecs)[0]

        scored: List[Tuple[ExperimentRecord, float]] = list(zip(records, sims.tolist()))
        if physics_domain:
            domain_q = physics_domain.lower()
            boosted = []
            for rec, sim in scored:
                domain_r = (rec.physics_domain or "").lower()
                boost = 0.0
                if domain_r == domain_q:
                    boost = 0.25
                elif domain_r and (domain_q in domain_r or domain_r in domain_q):
                    boost = 0.1
                boosted.append((rec, min(1.0, sim + boost)))
            scored = boosted

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def get_lessons_for_domain(self, physics_domain: str) -> List[str]:
        """Every non-empty ``lessons_learned`` string logged under a
        ``physics_domain`` that case-insensitively matches ``physics_domain``,
        aggregated from the real stored records (most recent last, in the
        order ``ExperimentStore.list_runs`` returns them -- ascending by
        ``started_at`` within that domain)."""
        domain_q = physics_domain.lower()
        return [
            r.lessons_learned
            for r in self._all_records()
            if r.lessons_learned and (r.physics_domain or "").lower() == domain_q
        ]
