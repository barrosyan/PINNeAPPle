"""ANSYS Fluent live-solver coupling: launches a real ``fluent`` batch-mode
process (journal-file driven, ``fluent <precision> -g -i run.jou``) and
drives an actual simulation, then hands the result off to the existing
``cfd_formats.fluent_mesh_reader`` UPD bridge.

This is distinct from -- and complements -- ``cfd_formats.fluent_mesh_reader``,
which only reads ``.msh``/``.cas`` files Fluent has already produced and
has no live-solver coupling of its own. See ``runner.py``'s module
docstring for the honesty note on what could and couldn't be tested
locally (no licensed Fluent installation is available in this
environment).
"""
from .journal_builder import FluentCaseConfig, build_journal
from .runner import FluentRunConfig, FluentRunResult, run_fluent_case

__all__ = [
    "FluentCaseConfig",
    "build_journal",
    "FluentRunConfig",
    "FluentRunResult",
    "run_fluent_case",
]
