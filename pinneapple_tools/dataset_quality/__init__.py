"""pinneapple_tools.dataset_quality — dataset quality assessment.

Two independent ways to vet a dataset before training on it:

``checks``
    Generic, statistical/heuristic checks over a plain ``(coords,
    fields)`` point-cloud representation: completeness (NaN fraction),
    cross-field shape/monotonicity consistency, distribution summaries,
    IsolationForest outlier detection, physical-range issues. No
    framework dependency, no model involved.

``vlm_curation``
    Vision-language-model curation: run a VLM over a short clip of each
    sequence and use its label/confidence judgment (parsed from real
    generated text, never fabricated) to filter a raw dataset down to a
    curated one before training. Ported and generalized from the
    downstream ``physcurator`` project. Requires the ``vlm_curation``
    extra to actually run a model; the parsing/filtering logic itself has
    no heavy dependency.
"""
from .checks import (
    analyze_completeness,
    validate_consistency,
    analyze_distribution,
    detect_outliers,
    field_range_issues,
    build_1d_linear_interpolant,
)
from .vlm_curation import (
    DEFAULT_LABELS,
    CurationRecord,
    build_consistency_prompt,
    extract_json,
    normalize_label,
    clamp_confidence,
    parse_curation_response,
    load_vlm,
    curate_sequence,
    curate_dataset,
    filter_curated,
)

__all__ = [
    # checks
    "analyze_completeness",
    "validate_consistency",
    "analyze_distribution",
    "detect_outliers",
    "field_range_issues",
    "build_1d_linear_interpolant",
    # vlm_curation
    "DEFAULT_LABELS",
    "CurationRecord",
    "build_consistency_prompt",
    "extract_json",
    "normalize_label",
    "clamp_confidence",
    "parse_curation_response",
    "load_vlm",
    "curate_sequence",
    "curate_dataset",
    "filter_curated",
]
