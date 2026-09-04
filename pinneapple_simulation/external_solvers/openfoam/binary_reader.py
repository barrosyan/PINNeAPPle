"""Binary-format OpenFOAM FoamFile primitives.

``field_reader.py``'s ``_read_internal_field`` only understands ASCII
FoamFiles (a regex over UTF-8 text). Most real OpenFOAM cases -- anything
run with the (very common) ``writeFormat binary;`` control-dict setting --
cannot be read by it at all: a binary field/mesh file's payload is a raw
little-endian byte blob, not text, so the ASCII regexes either raise
``ValueError`` or silently mis-parse. There is also no reader anywhere in
PINNeAPPle for ``constant/polyMesh`` (points/faces/owner/neighbour), binary
or ASCII, so cell connectivity/positions could not be recovered at all
without an OpenFOAM installation's own ``writeCellCentres`` function
object having been run.

This module implements just enough of the binary (and, where a field
happens to be hand-written or exported as ASCII despite the case's
declared format, ASCII) FoamFile layout to read what a training pipeline
actually needs:

- scalar / vector / symmTensor / tensor volField ``internalField`` blocks
  (uniform and nonuniform, in either format).
- ``labelList`` (``constant/polyMesh/owner``, ``neighbour``).
- ``vectorField`` (``constant/polyMesh/points``).
- ``faceCompactList`` (``constant/polyMesh/faces``): an offsets labelList
  followed by an indices labelList -- the default face-storage format
  since OpenFOAM 4.

Binary layout (``arch "LSB;label=32;scalar=64"``, the default and by far
the most common OpenFOAM build configuration): a decimal ASCII count,
then ``(``, then the raw little-endian payload (no separators between
elements), then ``)``. Only 32-bit label / 64-bit scalar, little-endian
archives are supported; a big-endian or ``label=64`` build is out of
scope for this module.

See ``mesh_reader.py`` for cell-center/volume reconstruction built on top
of this module's polyMesh primitives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_LABEL_DTYPE = np.int32
_SCALAR_DTYPE = np.float64

_CLASS_RE = re.compile(rb"class\s+(\w+);")
_FORMAT_RE = re.compile(rb"format\s+(\w+);")


@dataclass
class FoamFileHeader:
    klass: str
    fmt: str
    body_start: int


def read_header(data: bytes) -> FoamFileHeader:
    """Locate the end of the ``FoamFile {...}`` block and parse its keys."""
    m = re.search(rb"FoamFile\s*\n?\{", data)
    if not m:
        raise ValueError("not an OpenFOAM FoamFile (no 'FoamFile {' block found)")
    brace_start = data.index(b"{", m.start())
    depth = 0
    i = brace_start
    while i < len(data):
        if data[i : i + 1] == b"{":
            depth += 1
        elif data[i : i + 1] == b"}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    header_block = data[brace_start : i + 1]
    cm = _CLASS_RE.search(header_block)
    fm = _FORMAT_RE.search(header_block)
    klass = cm.group(1).decode() if cm else ""
    fmt = fm.group(1).decode() if fm else "ascii"
    # Body starts after the closing '}' of FoamFile, skipping the
    # "// * * * ... * //" divider line if present.
    rest = data[i + 1 :]
    dm = re.match(rb"\s*//[^\n]*\n", rest)
    body_start = i + 1 + (dm.end() if dm else 0)
    return FoamFileHeader(klass=klass, fmt=fmt, body_start=body_start)


def _skip_ws(data: bytes, pos: int) -> int:
    n = len(data)
    while pos < n and data[pos : pos + 1] in b" \t\r\n":
        pos += 1
    return pos


def _read_int(data: bytes, pos: int) -> Tuple[int, int]:
    pos = _skip_ws(data, pos)
    start = pos
    n = len(data)
    while pos < n and (data[pos : pos + 1].isdigit() or data[pos : pos + 1] == b"-"):
        pos += 1
    if start == pos:
        raise ValueError(f"expected integer at byte offset {pos}, found {data[pos:pos+20]!r}")
    return int(data[start:pos]), pos


def read_binary_block(
    data: bytes, pos: int, n_components: int, dtype=_SCALAR_DTYPE
) -> Tuple[np.ndarray, int]:
    """Read ``<count>\\n(<raw bytes>)`` starting at *pos*.

    Returns the array (shape ``(count,)`` if ``n_components == 1`` else
    ``(count, n_components)``) and the byte offset just past the closing
    ``)``.
    """
    count, pos = _read_int(data, pos)
    pos = _skip_ws(data, pos)
    if data[pos : pos + 1] != b"(":
        raise ValueError(f"expected '(' at offset {pos}, found {data[pos:pos+20]!r}")
    pos += 1
    itemsize = np.dtype(dtype).itemsize
    nbytes = count * n_components * itemsize
    buf = data[pos : pos + nbytes]
    if len(buf) != nbytes:
        raise ValueError(f"truncated binary block: expected {nbytes} bytes, got {len(buf)}")
    arr = np.frombuffer(buf, dtype=dtype).copy()  # buf is a read-only slice of the source bytes
    if n_components > 1:
        arr = arr.reshape(count, n_components)
    pos += nbytes
    pos = _skip_ws(data, pos)
    if data[pos : pos + 1] != b")":
        raise ValueError(f"expected ')' at offset {pos}, found {data[pos:pos+20]!r}")
    pos += 1
    return arr, pos


def read_label_list(data: bytes, path_hint: str = "") -> np.ndarray:
    hdr = read_header(data)
    if hdr.fmt != "binary":
        raise NotImplementedError(f"{path_hint}: only binary format is supported, got {hdr.fmt!r}")
    arr, _ = read_binary_block(data, hdr.body_start, 1, dtype=_LABEL_DTYPE)
    return arr


def read_vector_field_points(data: bytes, path_hint: str = "") -> np.ndarray:
    hdr = read_header(data)
    if hdr.fmt != "binary":
        raise NotImplementedError(f"{path_hint}: only binary format is supported, got {hdr.fmt!r}")
    arr, _ = read_binary_block(data, hdr.body_start, 3, dtype=_SCALAR_DTYPE)
    return arr


def read_face_compact_list(data: bytes, path_hint: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Read a ``faceCompactList``: an offsets array (len nFaces+1) followed
    by a flat point-index array (len offsets[-1])."""
    hdr = read_header(data)
    if hdr.fmt != "binary":
        raise NotImplementedError(f"{path_hint}: only binary format is supported, got {hdr.fmt!r}")
    offsets, pos = read_binary_block(data, hdr.body_start, 1, dtype=_LABEL_DTYPE)
    indices, _ = read_binary_block(data, pos, 1, dtype=_LABEL_DTYPE)
    return offsets, indices


_FIELD_KIND = {
    "vector": ("volVectorField", 3),
    "scalar": ("volScalarField", 1),
    "symmTensor": ("volSymmTensorField", 6),
    "tensor": ("volTensorField", 9),
    "sphericalTensor": ("volSphericalTensorField", 1),
}


def _read_ascii_list_body(data: bytes, pos: int, n_components: int) -> Tuple[np.ndarray, int]:
    """Read ``<count>\\n(<ascii vectors/scalars>)`` starting at *pos*.

    Handles scalar lists (``1\\n2\\n``) and vector/tensor lists
    (``(1 2 3)\\n(4 5 6)``) alike by paren-depth-matching the outer list,
    then stripping inner parens before a vectorised float parse.
    """
    count, pos = _read_int(data, pos)
    pos = _skip_ws(data, pos)
    if data[pos : pos + 1] != b"(":
        raise ValueError(f"expected '(' at offset {pos}, found {data[pos:pos+20]!r}")
    start = pos + 1
    depth = 1
    i = start
    n = len(data)
    while i < n and depth > 0:
        c = data[i : i + 1]
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
        i += 1
    body = data[start : i - 1]
    cleaned = body.replace(b"(", b" ").replace(b")", b" ")
    nums = np.fromstring(cleaned, dtype=_SCALAR_DTYPE, sep=" ")
    expected = count * n_components
    if nums.size != expected:
        raise ValueError(f"ascii list parse mismatch: expected {expected} numbers, got {nums.size}")
    arr = nums.reshape(count, n_components) if n_components > 1 else nums
    return arr, i


def read_internal_field(
    data: bytes, path_hint: str = ""
) -> Tuple[np.ndarray, bool, Optional[int]]:
    """Read the ``internalField`` entry of a vol{Scalar,Vector,SymmTensor,...}Field.

    Returns ``(values, is_uniform, n_components)``. When uniform,
    ``values`` has shape ``(n_components,)`` (or a 1-element array for a
    scalar) and the caller must broadcast it to the mesh's cell count
    itself (the field file alone does not carry the cell count).

    A field file's own ``FoamFile.format`` can differ from the case's
    ``controlDict`` default -- e.g. a hand-written initial-condition field
    generated by a Python pre-processing script is commonly ASCII even
    when the solver writes binary for every later time -- so the format is
    read per-file from its own header, never assumed from the case.
    Matching is scoped to the ``internalField ... ;`` statement specifically
    (not the whole file), since a boundaryField patch can carry its own
    ``value uniform (...);`` entry that a whole-file search would wrongly
    latch onto first.
    """
    hdr = read_header(data)
    m = re.search(rb"internalField\s+(uniform|nonuniform)", data)
    if not m:
        raise ValueError(f"{path_hint}: 'internalField' not found")
    kind = m.group(1)
    pos = m.end()
    if kind == b"uniform":
        pos = _skip_ws(data, pos)
        end = data.index(b";", pos)
        raw = data[pos:end].strip()
        if raw.startswith(b"("):
            nums = [float(x) for x in raw.strip(b"()").split()]
            return np.array(nums, dtype=_SCALAR_DTYPE), True, len(nums)
        return np.array([float(raw)], dtype=_SCALAR_DTYPE), True, 1

    lm = re.match(rb"\s*List<(\w+)>", data[pos:])
    if not lm:
        raise ValueError(f"{path_hint}: expected 'List<type>' after 'nonuniform'")
    type_name = lm.group(1).decode()
    if type_name not in _FIELD_KIND:
        raise NotImplementedError(f"{path_hint}: unsupported list type List<{type_name}>")
    n_components = _FIELD_KIND[type_name][1]
    pos = pos + lm.end()
    if hdr.fmt == "binary":
        arr, _ = read_binary_block(data, pos, n_components, dtype=_SCALAR_DTYPE)
    elif hdr.fmt == "ascii":
        arr, _ = _read_ascii_list_body(data, pos, n_components)
    else:
        raise NotImplementedError(f"{path_hint}: unsupported FoamFile format {hdr.fmt!r}")
    return arr, False, n_components
