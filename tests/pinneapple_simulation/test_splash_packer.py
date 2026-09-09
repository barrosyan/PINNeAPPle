"""Tests for pinneapple_simulation.external_solvers.openfoam.splash_packer.

No .splash writer existed anywhere in this codebase before this module
(confirmed by a full-repo search before writing it) -- these tests cover
the packer, manifest reader, and integrity verifier against a small fake
OpenFOAM-shaped case directory (no real OpenFOAM install needed; the
packer only deals with files on disk, not their CFD content).
"""
from __future__ import annotations

import json
import os
import zipfile

import pytest

from pinneapple_simulation.external_solvers.openfoam.splash_packer import (
    pack_splash_archive,
    read_splash_manifest,
    verify_splash_archive,
)


def _make_fake_case(tmp_path, with_processor_dir=True):
    case_dir = tmp_path / "fake_case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir(parents=True)
    (case_dir / "0").mkdir(parents=True)
    (case_dir / "system" / "controlDict").write_text("application pimpleFoam;\n")
    (case_dir / "constant" / "transportProperties").write_text("nu 0.01;\n")
    (case_dir / "0" / "U").write_text("internalField uniform (0 0 0);\n")
    (case_dir / "case.foam").write_text("")  # ParaView placeholder, should be excluded by default
    if with_processor_dir:
        (case_dir / "processor0").mkdir(parents=True)
        (case_dir / "processor0" / "U").write_text("should not be packed\n")
    return str(case_dir)


def test_pack_creates_valid_zip_with_manifest(tmp_path):
    case_dir = _make_fake_case(tmp_path)
    out_path = str(tmp_path / "out.splash")

    manifest = pack_splash_archive(case_dir, out_path, summary={"solver": "pimpleFoam", "cells": 1000})

    assert os.path.exists(out_path)
    assert zipfile.is_zipfile(out_path)
    assert manifest["summary"] == {"solver": "pimpleFoam", "cells": 1000}
    assert manifest["case_name"] == "fake_case"
    assert "system/controlDict" in manifest["files"]
    assert "constant/transportProperties" in manifest["files"]
    assert "0/U" in manifest["files"]


def test_default_exclude_skips_processor_dirs_and_case_foam(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=True)
    out_path = str(tmp_path / "out.splash")
    manifest = pack_splash_archive(case_dir, out_path)

    assert not any(f.startswith("processor0") for f in manifest["files"])
    assert "case.foam" not in manifest["files"]
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        assert not any(n.startswith("processor0") for n in names)
        assert "case.foam" not in names


def test_manifest_sizes_and_hashes_are_correct(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")
    manifest = pack_splash_archive(case_dir, out_path)

    real_size = os.path.getsize(os.path.join(case_dir, "system", "controlDict"))
    assert manifest["files"]["system/controlDict"]["size"] == real_size

    import hashlib
    with open(os.path.join(case_dir, "system", "controlDict"), "rb") as fh:
        expected_sha = hashlib.sha256(fh.read()).hexdigest()
    assert manifest["files"]["system/controlDict"]["sha256"] == expected_sha


def test_read_splash_manifest_matches_pack_result(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")
    written = pack_splash_archive(case_dir, out_path)
    read_back = read_splash_manifest(out_path)
    assert read_back == written


def test_verify_splash_archive_clean_archive_has_no_issues(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")
    pack_splash_archive(case_dir, out_path)
    issues = list(verify_splash_archive(out_path))
    assert issues == []


def test_verify_splash_archive_detects_tampered_content(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")
    pack_splash_archive(case_dir, out_path)

    # Rewrite the archive with one file's content changed but the manifest
    # (and its recorded hash/size) left as originally written.
    tampered_path = str(tmp_path / "tampered.splash")
    with zipfile.ZipFile(out_path) as src, zipfile.ZipFile(tampered_path, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "0/U":
                data = b"internalField uniform (999 999 999);\n"  # tampered, different length too
            dst.writestr(item, data)

    issues = list(verify_splash_archive(tampered_path))
    assert any("0/U" in issue for issue in issues)


def test_verify_splash_archive_detects_missing_file(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")
    pack_splash_archive(case_dir, out_path)

    truncated_path = str(tmp_path / "truncated.splash")
    with zipfile.ZipFile(out_path) as src, zipfile.ZipFile(truncated_path, "w") as dst:
        for item in src.infolist():
            if item.filename == "0/U":
                continue  # drop this file, but its manifest entry stays
            dst.writestr(item, src.read(item.filename))

    issues = list(verify_splash_archive(truncated_path))
    assert any("0/U" in issue and "missing" in issue for issue in issues)


def test_custom_exclude_function(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")

    manifest = pack_splash_archive(case_dir, out_path, exclude=lambda rel: rel.endswith("transportProperties"))
    assert "constant/transportProperties" not in manifest["files"]
    assert "system/controlDict" in manifest["files"]


def test_case_name_defaults_to_dirname_but_can_be_overridden(tmp_path):
    case_dir = _make_fake_case(tmp_path, with_processor_dir=False)
    out_path = str(tmp_path / "out.splash")

    manifest = pack_splash_archive(case_dir, out_path, case_name="my_custom_case")
    assert manifest["case_name"] == "my_custom_case"
