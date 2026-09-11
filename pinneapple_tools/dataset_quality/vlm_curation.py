"""Vision-language-model dataset curation.

Generic engine for curating a synthetic training dataset by running a
vision-language model (VLM) over a short video clip of each sequence and
using its judgment of physical plausibility to filter a raw dataset down
to a "curated" one before training on it.

Ported and generalized from a downstream project (``physcurator``), which
used this exact approach -- clip -> VLM judgment -> JSON label/confidence
-> filter -- to screen synthetic 2D heat-diffusion sequences for
physically-implausible injected anomalies, against a fixed model
(``nvidia/Cosmos-Reason2-2B``) and a hardcoded heat-diffusion prompt.
Nothing in this module is heat2d-specific or Cosmos-specific: the VLM
model name/class, the prompt template (via a caller-supplied physics
description), and the raw-sequence -> clip-path lookup are all parameters
here. ``physcurator`` itself now calls into this module rather than
duplicating the logic locally.

Design, stated plainly (same "no fabricated results" stance as the rest of
this package): this module never invents a curation judgment. Every
label/confidence/reason returned by :func:`curate_sequence` /
:func:`curate_dataset` comes from parsing the VLM's own generated text
(via :func:`extract_json`); a response that is missing or isn't valid,
parseable JSON degrades to a neutral fallback label at confidence 0.5, it
is never silently promoted to a "consistent"/"passed" judgment. Running
:func:`curate_dataset` end to end against a real model requires the
``vlm_curation`` extra (``transformers`` + ``accelerate``) and, for a
hosted checkpoint such as ``nvidia/Cosmos-Reason2-2B``, a real GPU and a
Hugging Face download -- this module has no offline stand-in for the VLM's
judgment itself. :func:`curate_sequence` and :func:`curate_dataset` both
accept an already-loaded ``model``/``processor`` pair, so callers (and
tests) can inject a stub that never touches the network.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_LABELS: Tuple[str, ...] = ("consistent", "suspect", "inconsistent")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_consistency_prompt(
    pde_description: str,
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    extra_instructions: Optional[str] = None,
) -> str:
    """Build the text prompt asking a VLM to judge a clip's physical
    consistency against a stated governing-physics assumption.

    Parameters
    ----------
    pde_description : a short, physics-grounded statement of what the clip
        is supposed to show, e.g. ``"governing PDE is dT/dt =
        alpha * Laplacian(T), with constant Dirichlet boundary
        temperature"`` (the exact text ``physcurator`` uses for its
        heat-2D scenario) -- this is the only domain-specific input the
        prompt needs.
    labels : the closed label set the VLM must choose from. Defaults to
        ``("consistent", "suspect", "inconsistent")``.
    extra_instructions : optional additional free-text appended to the
        task instructions (e.g. dataset-specific caveats).

    Returns
    -------
    The full prompt string, unchanged in structure from ``physcurator``'s
    original ``curator/prompts.py::build_prompt`` (same three-part task,
    same "JSON only" instruction), just parameterized.
    """
    label_list = ", ".join(labels)
    example_label = labels[1] if len(labels) > 1 else labels[0]
    lines = [
        "You are a physical data curator for a synthetic scientific simulation clip.",
        f"Assumption: {pde_description.strip()}",
        "",
        "Task:",
        f"1) Classify the clip as one of: {label_list}.",
        "2) Provide a short physics-grounded reason.",
        "3) Provide confidence in [0,1].",
        "",
        "Return ONLY valid JSON with keys: label, reason, confidence.",
        (
            'Example: {"label":"%s","reason":"boundary condition jumps abruptly '
            'without a physically consistent transition","confidence":0.78}'
        )
        % example_label,
    ]
    if extra_instructions:
        lines.append(extra_instructions.strip())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Response parsing (pure, no model/framework dependency)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the first ``{...}`` JSON object found in
    a block of free-form generated text. Returns ``None`` (never raises)
    if no valid JSON object is present."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def normalize_label(
    label: Any,
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    fallback: Optional[str] = None,
) -> str:
    """Normalize a (possibly malformed) label value onto the closed
    ``labels`` set: exact match (case/whitespace-insensitive) first, then
    substring containment against each candidate label, else ``fallback``
    (defaults to the middle label of ``labels``, matching ``physcurator``'s
    original "suspect" -- neither pass nor fail -- default)."""
    if fallback is None:
        fallback = labels[len(labels) // 2] if labels else "suspect"
    if not isinstance(label, str):
        return fallback
    s = label.strip().lower()
    for cand in labels:
        if s == cand.lower():
            return cand
    # Substring containment, longest label first -- e.g. "inconsistent"
    # must be checked before "consistent", since the latter is a literal
    # substring of the former and would otherwise match first.
    for cand in sorted(labels, key=len, reverse=True):
        if cand.lower() in s or s in cand.lower():
            return cand
    return fallback


def clamp_confidence(x: Any, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce ``x`` to a float and clamp it to ``[lo, hi]``; a value that
    can't be parsed as a float falls back to the midpoint."""
    try:
        v = float(x)
    except Exception:
        v = (lo + hi) / 2.0
    return max(lo, min(hi, v))


@dataclass
class CurationRecord:
    """One sequence's curation judgment."""

    id: str
    label: str
    confidence: float
    reason: str
    raw_text: str = ""

    def to_dict(self, *, include_raw_text: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_raw_text:
            d.pop("raw_text", None)
        return d


def parse_curation_response(
    seq_id: str,
    text: str,
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    reason_max_chars: int = 300,
    raw_text_max_chars: int = 1200,
) -> CurationRecord:
    """Parse one VLM generation into a :class:`CurationRecord`. Never
    raises: a response that fails to parse as JSON yields the fallback
    label at confidence 0.5, with an empty reason -- the honest signal
    that the model's output was unusable, not a guessed judgment."""
    parsed = extract_json(text) or {}
    label = normalize_label(parsed.get("label"), labels=labels)
    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    confidence = clamp_confidence(parsed.get("confidence", 0.5))
    return CurationRecord(
        id=seq_id,
        label=label,
        confidence=confidence,
        reason=reason[:reason_max_chars],
        raw_text=text[:raw_text_max_chars],
    )


# ---------------------------------------------------------------------------
# Model loading (lazy -- transformers/accelerate are the optional
# 'vlm_curation' extra, not a core PINNeAPPle dependency)
# ---------------------------------------------------------------------------

def load_vlm(
    model_name: str,
    *,
    device: Optional[str] = None,
    dtype: Optional[Any] = None,
    model_cls: Optional[Any] = None,
    attn_implementation: str = "sdpa",
):
    """Load a Hugging Face vision-language chat model + its processor.

    Requires the ``vlm_curation`` extra (``pip install
    "pinneapple[vlm_curation]"`` -- ``transformers`` + ``accelerate``);
    raises a clear ``ImportError`` naming it if missing, same pattern as
    every other optional heavy-dependency bridge in this package (see
    ``pinneapple_llm.finetune.finetune_lora``).

    Parameters
    ----------
    model_name : a Hugging Face model id, e.g. ``"nvidia/Cosmos-Reason2-2B"``.
    model_cls : the model class to instantiate. Defaults to
        ``transformers.AutoModelForImageTextToText``, which resolves most
        chat-style VLMs; some checkpoints (Cosmos-Reason2 among them, which
        is a Qwen3-VL-architecture model) are not yet wired into that Auto
        class in every ``transformers`` release -- pass e.g.
        ``transformers.Qwen3VLForConditionalGeneration`` explicitly if the
        Auto class raises for your checkpoint.

    Returns
    -------
    ``(model, processor)``
    """
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError(
            "pinneapple_tools.dataset_quality.vlm_curation requires the 'vlm_curation' extra: "
            'pip install "pinneapple[vlm_curation]" (transformers, accelerate).'
        ) from e

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = dtype or (torch.bfloat16 if resolved_device == "cuda" else torch.float32)
    cls = model_cls or getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:  # pragma: no cover - only on very old transformers
        raise ImportError(
            "This transformers version has no AutoModelForImageTextToText; "
            "pass model_cls=<the checkpoint's real model class> explicitly."
        )

    model = cls.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

def curate_sequence(
    model: Any,
    processor: Any,
    clip_path: str,
    prompt: str,
    *,
    seq_id: str = "",
    fps: int = 4,
    max_new_tokens: int = 512,
    labels: Sequence[str] = DEFAULT_LABELS,
    system_message: str = "You are a helpful assistant.",
) -> CurationRecord:
    """Run one VLM generation over a single clip and parse the result.

    ``model``/``processor`` are duck-typed to the Hugging Face chat-VLM
    contract used by ``processor.apply_chat_template`` +
    ``model.generate`` + ``processor.batch_decode`` (exactly what
    :func:`load_vlm` returns) -- this function itself imports no
    heavy dependency, so it can be exercised in tests against a stub
    model/processor pair with no ``transformers``/GPU involved.
    """
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{os.path.abspath(clip_path)}", "fps": fps},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parse_curation_response(seq_id, text, labels=labels)


def curate_dataset(
    index: Sequence[Mapping[str, Any]],
    *,
    out_dir: str,
    model_name: str = "nvidia/Cosmos-Reason2-2B",
    model: Any = None,
    processor: Any = None,
    model_cls: Optional[Any] = None,
    prompt: Optional[str] = None,
    pde_description: Optional[str] = None,
    labels: Sequence[str] = DEFAULT_LABELS,
    id_field: str = "id",
    clip_field: str = "mp4",
    raw_sequence_loader: Optional[Callable[[Mapping[str, Any]], str]] = None,
    fps: int = 4,
    max_new_tokens: int = 512,
    resume: bool = True,
    device: Optional[str] = None,
) -> List[CurationRecord]:
    """Curate every sequence in ``index`` with a VLM and write per-sequence
    + aggregate curation metadata to ``out_dir`` (mirrors
    ``physcurator``'s ``curator/run_cosmos.py`` orchestration, generalized).

    Parameters
    ----------
    index : one entry per sequence, each a mapping with at least an
        ``id_field`` key and whatever the ``raw_sequence_loader`` (or,
        by default, ``clip_field``) needs to locate that sequence's clip.
    raw_sequence_loader : ``item -> clip_path``. Defaults to
        ``lambda item: item[clip_field]`` (the common case: the index
        entry already carries a path to a pre-rendered clip, as
        ``physcurator``'s ``data/generate_heat2d.py`` produces).
    model, processor : an already-loaded VLM + processor pair (see
        :func:`load_vlm`). If either is ``None``, both are loaded via
        ``load_vlm(model_name, model_cls=model_cls, device=device)`` --
        requires the ``vlm_curation`` extra and, for a real checkpoint, a
        GPU.
    model_cls : forwarded to :func:`load_vlm` when ``model``/``processor``
        aren't already loaded -- pass this when ``model_name``'s
        checkpoint needs a specific model class rather than
        ``transformers.AutoModelForImageTextToText`` (e.g.
        ``nvidia/Cosmos-Reason2-2B`` needs
        ``transformers.Qwen3VLForConditionalGeneration`` on at least some
        ``transformers`` releases -- see :func:`load_vlm`'s docstring).
    prompt : the full prompt text. If not given, built via
        :func:`build_consistency_prompt` from ``pde_description``
        (required in that case).
    resume : skip sequences that already have both a per-sequence JSON
        file in ``out_dir`` and an entry in ``out_dir/curation_index.json``
        -- lets a long curation run be safely re-launched.

    Returns
    -------
    The full list of :class:`CurationRecord` (previously-done + freshly
    curated), sorted by id -- also the content written to
    ``out_dir/curation_index.json``.
    """
    if prompt is None:
        if pde_description is None:
            raise ValueError("curate_dataset needs either `prompt` or `pde_description`.")
        prompt = build_consistency_prompt(pde_description, labels=labels)

    if raw_sequence_loader is None:
        raw_sequence_loader = lambda item: item[clip_field]  # noqa: E731

    if model is None or processor is None:
        model, processor = load_vlm(model_name, device=device, model_cls=model_cls)

    os.makedirs(out_dir, exist_ok=True)
    curation_index_path = os.path.join(out_dir, "curation_index.json")

    records: Dict[str, CurationRecord] = {}
    if resume and os.path.exists(curation_index_path):
        with open(curation_index_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                records[r["id"]] = CurationRecord(
                    id=r["id"], label=r["label"], confidence=r["confidence"],
                    reason=r.get("reason", ""),
                )

    for item in index:
        seq_id = item[id_field]
        out_path = os.path.join(out_dir, f"{seq_id}.json")
        if resume and seq_id in records and os.path.exists(out_path):
            continue

        clip_path = raw_sequence_loader(item)
        record = curate_sequence(
            model, processor, clip_path, prompt,
            seq_id=seq_id, fps=fps, max_new_tokens=max_new_tokens, labels=labels,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        records[seq_id] = record
        with open(curation_index_path, "w", encoding="utf-8") as f:
            json.dump(
                [records[k].to_dict(include_raw_text=False) for k in sorted(records)],
                f, indent=2,
            )

    return [records[k] for k in sorted(records)]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_curated(
    index: Sequence[Mapping[str, Any]],
    curation_records: Sequence[Any],
    *,
    keep_labels: Sequence[str] = ("consistent",),
    min_confidence: float = 0.55,
    id_field: str = "id",
    data_field: str = "npz",
    out_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter ``index`` down to entries whose curation judgment is in
    ``keep_labels`` and meets ``min_confidence`` (mirrors ``physcurator``'s
    ``data/make_curated.py``, generalized off the ``"npz"`` field name and
    off filesystem copying being mandatory).

    ``curation_records`` may be :class:`CurationRecord` instances or plain
    dicts with the same keys (as read back from a ``curation_index.json``).

    If ``out_dir`` is given, the kept entries' ``data_field`` file is
    copied into it (``shutil.copy2``) and the returned entry's
    ``data_field`` is rewritten to that new path -- matching
    ``make_curated.py``'s behaviour of materializing a standalone curated
    dataset directory. If ``out_dir`` is ``None``, filtering only: no
    filesystem I/O, ``data_field`` is left untouched.

    Returns the kept entries (each the original index item plus
    ``label``/``confidence``/``reason``), and, if ``out_dir`` is given,
    also writes ``out_dir/index.json`` with that same list.
    """
    keep_set = set(keep_labels)
    by_id = {}
    for r in curation_records:
        rd = r.to_dict() if isinstance(r, CurationRecord) else dict(r)
        by_id[rd["id"]] = rd

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    kept: List[Dict[str, Any]] = []
    for item in index:
        cid = item[id_field]
        c = by_id.get(cid)
        if not c:
            continue
        if c["label"] not in keep_set or float(c["confidence"]) < min_confidence:
            continue

        entry = dict(item)
        entry["label"] = c["label"]
        entry["confidence"] = float(c["confidence"])
        entry["reason"] = c.get("reason", "")[:200]

        if out_dir is not None:
            import shutil

            src = item[data_field]
            dst = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            entry[data_field] = dst

        kept.append(entry)

    if out_dir is not None:
        with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2)

    return kept
