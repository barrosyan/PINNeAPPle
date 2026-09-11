"""Tests for pinneapple_tools.dataset_quality.vlm_curation.

Exercises the full pipeline -- prompt building, response parsing,
per-sequence curation, dataset-level orchestration (with resume), and
filtering -- without touching a real VLM checkpoint or GPU:
``curate_sequence``/``curate_dataset`` accept an already-loaded
``model``/``processor`` pair, so a duck-typed stub pair (mimicking the
Hugging Face chat-VLM ``apply_chat_template`` / ``generate`` /
``batch_decode`` contract with real ``torch`` tensors) stands in for the
real model. This tests the real orchestration/parsing/filtering logic
end to end; it does not and cannot verify that a real VLM (e.g.
``nvidia/Cosmos-Reason2-2B``) produces a *correct* physical judgment --
that requires the actual model and a GPU, well outside this test's scope.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from pinneapple_tools.dataset_quality.vlm_curation import (
    DEFAULT_LABELS,
    CurationRecord,
    build_consistency_prompt,
    clamp_confidence,
    curate_dataset,
    curate_sequence,
    extract_json,
    filter_curated,
    normalize_label,
    parse_curation_response,
)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def test_build_consistency_prompt_embeds_description_and_labels():
    prompt = build_consistency_prompt("dT/dt = alpha * Laplacian(T)")
    assert "dT/dt = alpha * Laplacian(T)" in prompt
    for label in DEFAULT_LABELS:
        assert label in prompt
    assert "JSON" in prompt


def test_build_consistency_prompt_custom_labels():
    prompt = build_consistency_prompt("mass is conserved", labels=("ok", "bad"))
    assert "ok" in prompt and "bad" in prompt
    assert "consistent" not in prompt.split("Assumption")[0]  # no leaked default label text


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------

def test_extract_json_finds_embedded_object():
    text = 'Some preamble.\n{"label": "consistent", "confidence": 0.9}\ntrailer'
    assert extract_json(text) == {"label": "consistent", "confidence": 0.9}


def test_extract_json_returns_none_for_no_json():
    assert extract_json("no json here at all") is None


def test_extract_json_returns_none_for_malformed_json():
    assert extract_json("{not: valid, json}") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("consistent", "consistent"),
        ("Consistent", "consistent"),
        ("  INCONSISTENT  ", "inconsistent"),
        ("this looks inconsistent to me", "inconsistent"),
        ("totally consistent behavior", "consistent"),
    ],
)
def test_normalize_label_matches_known_labels(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_falls_back_on_garbage():
    assert normalize_label("garbage") == "suspect"
    assert normalize_label(None) == "suspect"
    assert normalize_label(42) == "suspect"


def test_normalize_label_custom_fallback():
    assert normalize_label("garbage", labels=("ok", "bad"), fallback="ok") == "ok"


@pytest.mark.parametrize(
    "raw,expected",
    [(0.5, 0.5), (-1.0, 0.0), (2.0, 1.0), ("0.7", 0.7), ("nan-ish", 0.5)],
)
def test_clamp_confidence(raw, expected):
    assert clamp_confidence(raw) == pytest.approx(expected)


def test_parse_curation_response_valid_json():
    text = '{"label":"suspect","reason":"boundary jump","confidence":0.78}'
    rec = parse_curation_response("seq_0000", text)
    assert rec.id == "seq_0000"
    assert rec.label == "suspect"
    assert rec.reason == "boundary jump"
    assert rec.confidence == pytest.approx(0.78)


def test_parse_curation_response_unparseable_never_fabricates_a_pass():
    rec = parse_curation_response("seq_0001", "the model rambled without ever emitting JSON")
    assert rec.label == "suspect"  # neutral, not "consistent"
    assert rec.confidence == pytest.approx(0.5)
    assert rec.reason == ""


# ---------------------------------------------------------------------------
# Stub VLM (duck-typed to the HF chat-VLM contract) for curate_sequence /
# curate_dataset, with no transformers/GPU dependency.
# ---------------------------------------------------------------------------

class _FakeBatch(dict):
    """Mimics a HF ``BatchFeature``: dict-like (for **kwargs expansion),
    with a ``.to(device)`` and an ``.input_ids`` attribute."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _FakeBatch(torch.zeros((1, 5), dtype=torch.long))

    def batch_decode(self, sequences, **kwargs):
        return self._next_responses.pop(0)


class _FakeModel:
    device = "cpu"

    def __init__(self, responses):
        # each call to generate() consumes one canned response text
        self._responses = list(responses)
        self.generate_calls = 0

    def generate(self, **inputs):
        self.generate_calls += 1
        # 5 "input" tokens + 3 "new" tokens, matching input_ids shape (1, 5)
        return torch.zeros((1, 8), dtype=torch.long)


def _wire_fake_pair(response_texts):
    model = _FakeModel(response_texts)
    processor = _FakeProcessor()
    processor._next_responses = [[t] for t in response_texts]
    return model, processor


def test_curate_sequence_runs_fake_model_and_parses_response(tmp_path):
    clip = tmp_path / "seq_0000.mp4"
    clip.write_bytes(b"not a real video, just needs to exist as a path")

    model, processor = _wire_fake_pair(
        ['{"label":"consistent","reason":"smooth diffusion","confidence":0.92}']
    )
    rec = curate_sequence(model, processor, str(clip), "some prompt", seq_id="seq_0000")

    assert isinstance(rec, CurationRecord)
    assert rec.id == "seq_0000"
    assert rec.label == "consistent"
    assert rec.confidence == pytest.approx(0.92)
    assert model.generate_calls == 1
    # the clip path made it into the chat-template call as an absolute file:// URI
    messages, _ = processor.calls[0]
    video_content = messages[1]["content"][0]
    assert video_content["type"] == "video"
    assert video_content["video"] == f"file://{os.path.abspath(str(clip))}"


def _make_index(tmp_path, n=3):
    index = []
    for i in range(n):
        clip = tmp_path / f"seq_{i:04d}.mp4"
        clip.write_bytes(b"fake clip")
        index.append({"id": f"seq_{i:04d}", "mp4": str(clip), "npz": str(tmp_path / f"seq_{i:04d}.npz")})
        np.savez(index[-1]["npz"], frames=np.zeros((2, 2, 2), dtype=np.float32))
    return index


def test_curate_dataset_writes_per_sequence_and_index_metadata(tmp_path):
    index = _make_index(tmp_path, n=2)
    out_dir = tmp_path / "metadata"

    responses = [
        '{"label":"consistent","reason":"fine","confidence":0.9}',
        '{"label":"inconsistent","reason":"boundary jump","confidence":0.8}',
    ]
    model, processor = _wire_fake_pair(responses)

    records = curate_dataset(
        index,
        out_dir=str(out_dir),
        model=model,
        processor=processor,
        pde_description="dT/dt = alpha * Laplacian(T)",
        resume=False,
    )

    assert [r.id for r in records] == ["seq_0000", "seq_0001"]
    assert records[0].label == "consistent"
    assert records[1].label == "inconsistent"

    # per-sequence JSON files exist
    assert (out_dir / "seq_0000.json").exists()
    assert (out_dir / "seq_0001.json").exists()

    # aggregate curation_index.json exists and matches
    with open(out_dir / "curation_index.json") as f:
        agg = json.load(f)
    assert {a["id"] for a in agg} == {"seq_0000", "seq_0001"}
    # raw_text is intentionally excluded from the aggregate index
    assert all("raw_text" not in a for a in agg)


def test_curate_dataset_resume_skips_already_done_sequences(tmp_path):
    index = _make_index(tmp_path, n=2)
    out_dir = tmp_path / "metadata"

    # First pass: curate only seq_0000.
    model1, processor1 = _wire_fake_pair(
        ['{"label":"consistent","reason":"fine","confidence":0.9}']
    )
    curate_dataset(
        index[:1], out_dir=str(out_dir), model=model1, processor=processor1,
        pde_description="dT/dt = alpha * Laplacian(T)", resume=True,
    )
    assert model1.generate_calls == 1

    # Second pass over the full index with resume=True: only seq_0001 should
    # trigger a fresh model call.
    model2, processor2 = _wire_fake_pair(
        ['{"label":"suspect","reason":"unsure","confidence":0.6}']
    )
    records = curate_dataset(
        index, out_dir=str(out_dir), model=model2, processor=processor2,
        pde_description="dT/dt = alpha * Laplacian(T)", resume=True,
    )

    assert model2.generate_calls == 1  # only the missing sequence was (re)curated
    by_id = {r.id: r for r in records}
    assert by_id["seq_0000"].label == "consistent"  # preserved from pass 1
    assert by_id["seq_0001"].label == "suspect"     # freshly curated in pass 2


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_filter_curated_keeps_high_confidence_matching_labels_no_copy():
    index = [{"id": "a", "npz": "a.npz"}, {"id": "b", "npz": "b.npz"}, {"id": "c", "npz": "c.npz"}]
    records = [
        CurationRecord(id="a", label="consistent", confidence=0.9, reason="ok"),
        CurationRecord(id="b", label="consistent", confidence=0.3, reason="low confidence"),
        CurationRecord(id="c", label="inconsistent", confidence=0.95, reason="bad"),
    ]
    kept = filter_curated(index, records, keep_labels=("consistent",), min_confidence=0.55)
    assert [k["id"] for k in kept] == ["a"]
    assert kept[0]["npz"] == "a.npz"  # untouched -- no out_dir given


def test_filter_curated_copies_files_and_writes_index_when_out_dir_given(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    curated_dir = tmp_path / "curated"

    npz_path = raw_dir / "seq_0000.npz"
    np.savez(npz_path, frames=np.zeros((2, 2), dtype=np.float32))

    index = [{"id": "seq_0000", "npz": str(npz_path)}]
    records = [{"id": "seq_0000", "label": "consistent", "confidence": 0.8, "reason": "ok"}]

    kept = filter_curated(
        index, records, keep_labels=("consistent",), min_confidence=0.5, out_dir=str(curated_dir),
    )

    assert len(kept) == 1
    copied_path = curated_dir / "seq_0000.npz"
    assert copied_path.exists()
    assert kept[0]["npz"] == str(copied_path)

    with open(curated_dir / "index.json") as f:
        written = json.load(f)
    assert written == kept
