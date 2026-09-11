"""Tests for the real local-code-repository corpus source added to
``pinneapple_analysis.retrieval.corpus`` for org-management's knowledge-
base requirement (``code_repo_documents`` / ``iter_repo_files`` /
``chunk_text`` / ``file_fingerprint``).

No network, no sentence-transformers needed -- these exercise only real
filesystem I/O against a real temp directory this test creates, never a
mocked filesystem.
"""
from __future__ import annotations

import os

from pinneapple_analysis.retrieval.corpus import (
    Document,
    chunk_text,
    code_repo_documents,
    file_fingerprint,
    iter_repo_files,
)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_iter_repo_files_finds_real_files_and_skips_excluded_dirs(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "pkg", "mod.py"), "def f():\n    return 1\n")
    _write(os.path.join(root, "README.md"), "# hello\n")
    _write(os.path.join(root, "node_modules", "dep.js"), "module.exports = 1;\n")
    _write(os.path.join(root, ".git", "config.py"), "should_not_appear = True\n")
    _write(os.path.join(root, "data.bin"), "not a tracked extension\n".replace("py", "bin"))

    found = iter_repo_files(root)
    rels = {os.path.relpath(p, root) for p in found}
    assert "pkg/mod.py" in rels
    assert "README.md" in rels
    assert not any("node_modules" in r for r in rels)
    assert not any(r.startswith(".git") for r in rels)


def test_iter_repo_files_skips_oversized_files(tmp_path):
    root = str(tmp_path)
    big_path = os.path.join(root, "big.py")
    _write(big_path, "x = 1\n" * 100)
    small_path = os.path.join(root, "small.py")
    _write(small_path, "x = 1\n")

    found = iter_repo_files(root, max_file_bytes=50)
    rels = {os.path.relpath(p, root) for p in found}
    assert "small.py" in rels
    assert "big.py" not in rels  # real size check, not a fabricated skip


def test_iter_repo_files_never_returns_a_nonexistent_or_empty_path(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "empty.py"), "")
    _write(os.path.join(root, "real.py"), "x = 1\n")
    found = iter_repo_files(root)
    rels = {os.path.relpath(p, root) for p in found}
    assert "empty.py" not in rels  # zero-byte file excluded by the size check (0 < size)
    assert "real.py" in rels
    for p in found:
        assert os.path.isfile(p)


def test_chunk_text_returns_whole_text_when_it_fits():
    text = "short module docstring\nand a second line"
    chunks = chunk_text(text, max_chars=4000)
    assert chunks == [text]


def test_chunk_text_splits_long_text_and_every_chunk_is_a_verbatim_slice():
    paragraph = "line of real code\n" * 50  # deterministic, non-empty content
    text = "\n\n".join([paragraph] * 5)  # ~4750 chars, several blank-line boundaries
    chunks = chunk_text(text, max_chars=1000, overlap=50)
    assert len(chunks) > 1
    reassembled = "".join(chunks)
    # every chunk is real text that actually occurs in the source (no fabrication)
    for c in chunks:
        assert c in text
    assert reassembled  # never produces only empty chunks


def test_chunk_text_rejects_nonpositive_max_chars():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0)


def test_file_fingerprint_is_stable_and_changes_with_content(tmp_path):
    path = os.path.join(str(tmp_path), "f.py")
    _write(path, "x = 1\n")
    h1 = file_fingerprint(path)
    h2 = file_fingerprint(path)
    assert h1 == h2  # deterministic real hash, not a random/mocked value

    _write(path, "x = 2\n")
    h3 = file_fingerprint(path)
    assert h3 != h1  # real content change is really detected


def test_code_repo_documents_produces_real_verbatim_documents(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "core.py"), "def add(a, b):\n    return a + b\n")
    _write(os.path.join(root, "sub", "util.py"), "CONST = 42\n")

    docs = code_repo_documents(root, repo_name="demo-repo")
    assert docs, "should produce at least one real document"
    assert all(isinstance(d, Document) for d in docs)

    by_title = {d.title: d for d in docs}
    assert "demo-repo/core.py" in by_title
    core_doc = by_title["demo-repo/core.py"]
    assert "def add(a, b):" in core_doc.text  # verbatim, not a summary
    assert core_doc.source == os.path.join(root, "core.py")
    assert core_doc.doc_id == "code:demo-repo:core.py#0"

    assert "demo-repo/sub/util.py" in by_title


def test_code_repo_documents_chunks_a_large_file_into_multiple_documents(tmp_path):
    root = str(tmp_path)
    big_text = "\n\n".join(f"def f_{i}():\n    return {i}\n" for i in range(500))
    _write(os.path.join(root, "big.py"), big_text)

    docs = code_repo_documents(root, repo_name="demo-repo", max_chars_per_chunk=500)
    big_docs = [d for d in docs if d.title == "demo-repo/big.py"]
    assert len(big_docs) > 1
    doc_ids = [d.doc_id for d in big_docs]
    assert doc_ids == sorted(doc_ids, key=lambda s: int(s.rsplit("#", 1)[1]))
    assert len(doc_ids) == len(set(doc_ids))  # no duplicate chunk ids


def test_code_repo_documents_only_paths_restricts_the_scan(tmp_path):
    root = str(tmp_path)
    p1 = os.path.join(root, "a.py")
    p2 = os.path.join(root, "b.py")
    _write(p1, "a = 1\n")
    _write(p2, "b = 2\n")

    docs = code_repo_documents(root, repo_name="demo-repo", only_paths=[p1])
    titles = {d.title for d in docs}
    assert titles == {"demo-repo/a.py"}  # b.py never read, matching an incremental refresh's intent


def test_code_repo_documents_never_produces_duplicate_doc_ids(tmp_path):
    root = str(tmp_path)
    for i in range(10):
        _write(os.path.join(root, f"m{i}.py"), f"x = {i}\n")
    docs = code_repo_documents(root, repo_name="demo-repo")
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids))
