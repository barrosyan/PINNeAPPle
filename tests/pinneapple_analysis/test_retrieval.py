"""Tests for pinneapple_analysis.retrieval -- corpus assembly + real
embedding-based search (no mocked embeddings; this exercises the real
sentence-transformers model, matching this codebase's live-dependency
test convention for anything that needs a real local model/service).

Skipped (not failed) if the 'retrieval' extra (sentence-transformers)
isn't installed, matching the pattern used for other optional-extra
tests in this repo.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers", reason="'retrieval' extra not installed")

from pinneapple_analysis.retrieval.corpus import Document, build_corpus
from pinneapple_analysis.retrieval.index import RetrievalIndex, SearchResult


# ---------------------------------------------------------------------------
# corpus.py -- real content, no live embedding needed
# ---------------------------------------------------------------------------

def test_build_corpus_is_real_and_substantial():
    docs = build_corpus()
    assert len(docs) > 50  # 14 verification modules + 65 real presets, as of writing
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.text.strip() for d in docs)  # never an empty/placeholder document


def test_corpus_includes_known_verification_modules():
    docs = build_corpus()
    doc_ids = {d.doc_id for d in docs}
    assert "verification_module:architecture_critique" in doc_ids
    assert "verification_module:dimensional_analysis" in doc_ids


def test_corpus_includes_real_presets():
    docs = build_corpus()
    doc_ids = {d.doc_id for d in docs}
    assert "preset:burgers_1d" in doc_ids
    assert "preset:steady_heat_conduction_3d" in doc_ids


def test_corpus_documents_are_never_duplicated():
    docs = build_corpus()
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids))


# ---------------------------------------------------------------------------
# index.py -- real embeddings, real search (slower, real local model)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_index() -> RetrievalIndex:
    # A small, fixed, real (not fabricated) document set -- fast to embed,
    # covering distinct enough topics that a real semantic search should
    # separate them cleanly.
    docs = [
        Document(doc_id="d1", source="test", title="turbulence", text=(
            "Wall-bounded turbulent channel flow, WALE LES closure, real OpenFOAM simulation "
            "of velocity and pressure fields near a solid wall."
        )),
        Document(doc_id="d2", source="test", title="orbital mechanics", text=(
            "Two-body Kepler orbit, satellite position and velocity propagation around a "
            "central gravitating body under Newtonian gravity."
        )),
        Document(doc_id="d3", source="test", title="heat conduction", text=(
            "Steady-state heat conduction equation, temperature field, thermal diffusivity, "
            "Dirichlet and Neumann boundary conditions on a solid domain."
        )),
    ]
    return RetrievalIndex.build(docs)


def test_build_empty_corpus_raises():
    with pytest.raises(ValueError, match="nothing to index"):
        RetrievalIndex.build([])


def test_search_finds_the_real_semantic_match(small_index):
    results = small_index.search("wall shear stress in a turbulent boundary layer", top_k=1)
    assert len(results) == 1
    assert results[0].document.doc_id == "d1"


def test_search_finds_orbital_mechanics_match(small_index):
    results = small_index.search("satellite orbiting a planet under gravity", top_k=1)
    assert results[0].document.doc_id == "d2"


def test_search_returns_at_most_corpus_size(small_index):
    results = small_index.search("anything", top_k=100)
    assert len(results) == 3  # corpus only has 3 documents, never padded


def test_search_scores_are_ordered_descending(small_index):
    results = small_index.search("heat transfer through a solid material", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_rejects_non_positive_top_k(small_index):
    with pytest.raises(ValueError, match="must be positive"):
        small_index.search("x", top_k=0)


def test_save_and_load_round_trip_preserves_scores(small_index, tmp_path):
    path = str(tmp_path / "index.npz")
    small_index.save(path)
    loaded = RetrievalIndex.load(path)

    assert len(loaded.documents) == len(small_index.documents)
    assert {d.doc_id for d in loaded.documents} == {d.doc_id for d in small_index.documents}

    original = small_index.search("wall shear stress in a turbulent boundary layer", top_k=1)
    reloaded = loaded.search("wall shear stress in a turbulent boundary layer", top_k=1)
    assert reloaded[0].document.doc_id == original[0].document.doc_id
    assert reloaded[0].score == pytest.approx(original[0].score, abs=1e-4)


def test_index_constructor_rejects_mismatched_lengths():
    import numpy as np
    docs = [Document(doc_id="d1", source="x", title="x", text="x")]
    embeddings = np.zeros((2, 4), dtype=np.float32)  # 2 rows for 1 document
    with pytest.raises(ValueError, match="must match 1:1"):
        RetrievalIndex(documents=docs, embeddings=embeddings, model_name="fake")
