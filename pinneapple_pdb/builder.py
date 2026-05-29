"""Builder for physical datasets from Earth data hubs (NASA CMR, earthaccess)."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr

import earthaccess
from cmr import GranuleQuery

from .templates import schema_templates
from .validate import ValidationSpec, standardize_dims, validate_dataset
from .derived import DerivedSpec, apply_derived
from .shard import ShardSpec, iter_time_windows, subset_time, iter_tiles, subset_tile, regime_tags_for

# Default packs
VARIABLE_PACKS: Dict[str, List[str]] = {
    "core_state_2d": [
        "PS", "SLP", "TS",
        "T2M", "T2MDEW", "T2MWET",
        "U2M", "V2M",
        "U10M", "V10M",
        "U50M", "V50M",
        "QV2M", "QV10M",
        "TQV",
    ],
    "upper_air_plev": [
        "T250", "T500", "T850",
        "Q250", "Q500", "Q850",
        "U250", "U500", "U850",
        "V250", "V500", "V850",
        "OMEGA500",
        "H250", "H500", "H850", "H1000",
    ],
    "cloud_pbl": ["PBLTOP", "ZLCL", "CLDPRS", "CLDTMP", "TQL", "TQI", "DISPH"],
    "tropopause": ["TROPT", "TROPPT", "TROPQ", "TROPPB", "TROPPV"],
    "chemistry": ["TO3", "TOX"],
}

def _uid(txt: str) -> str:
    """Generate a 16-char hex hash from text for unique identifiers."""
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory (and parents) if missing; return Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_time(s: Optional[str]) -> Optional[str]:
    """Normalize time string to ISO format (YYYY-MM-DD -> YYYY-MM-DDT00:00:00Z)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # allow YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s}T00:00:00Z"
    # normalize ISO
    if s.endswith("Z"):
        return s
    return s  # user responsibility (kept simple for MVP)


def _temporal(time_start: Optional[str], time_end: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse and validate temporal range; returns (start, end) tuple or None."""
    if time_start is None and time_end is None:
        return None
    a = _normalize_time(time_start)
    b = _normalize_time(time_end)
    if not a or not b:
        raise ValueError("time_start e time_end precisam ser informados juntos.")
    return (a, b)


def _match(names: List[str], patterns: List[str]) -> List[str]:
    """Match names against patterns (glob or re:... prefix); returns sorted list of matches."""
    import re, fnmatch
    out = set()
    for pat in patterns or []:
        pat = (pat or "").strip()
        if not pat:
            continue
        if pat.startswith("re:"):
            rx = re.compile(pat[3:])
            for n in names:
                if rx.search(n):
                    out.add(n)
        elif any(ch in pat for ch in ["*", "?", "[", "]"]):
            for n in names:
                if fnmatch.fnmatch(n, pat):
                    out.add(n)
        else:
            if pat in names:
                out.add(pat)
    return sorted(out)


def _resolve_packs(packs: List[str], available: List[str], packs_dict: Dict[str, List[str]]) -> List[str]:
    """Resolve pack names to variable list, filtered by available; preserves order."""
    want: List[str] = []
    for p in packs or []:
        want += packs_dict.get(p, [])
    want = [v for v in want if v in available]
    seen=set(); out=[]
    for v in want:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


def _pick_url(granule: Any, prefer: str = "any") -> str:
    """Pick best URL from granule links (opendap, https, or any)."""
    prefer = (prefer or "any").lower()
    gd = getattr(granule, "data", granule)
    links = gd.get("links", []) or []
    if not links:
        raise RuntimeError("Granule sem links.")
    def score(lk: dict) -> int:
        href = (lk.get("href") or "").lower()
        title = (lk.get("title") or "").lower()
        rel = (lk.get("rel") or "").lower()
        is_opendap = ("opendap" in href) or ("opendap" in title) or ("dap" in title)
        is_https = href.startswith("https://") and ("data" in rel or "download" in title or href.endswith((".nc",".nc4",".h5",".hdf",".he5")))
        if prefer == "opendap":
            return 100 if is_opendap else (10 if is_https else 0)
        if prefer == "https":
            return 100 if is_https else (10 if is_opendap else 0)
        return 50 if (is_opendap or is_https) else 0
    best, best_sc = None, -1
    for lk in links:
        if not isinstance(lk, dict) or not lk.get("href"):
            continue
        sc = score(lk)
        if sc > best_sc:
            best_sc, best = sc, lk["href"]
    if not best:
        raise RuntimeError("Sem URL adequada.")
    return best

@dataclass
class HubQuery:
    """Query parameters for Earth data hub search."""

    provider: Optional[str] = None
    short_name: Optional[str] = None
    keyword: Optional[str] = None

@dataclass
class SpaceTime:
    """Temporal and spatial extent plus stride/chunking options."""

    time_start: Optional[str] = None
    time_end: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (W,S,E,N)
    stride: Dict[str, int] = field(default_factory=dict)
    chunking: Dict[str, int] = field(default_factory=dict)

@dataclass
class VariableSelection:
    """Variable include/exclude/pack selection."""

    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    packs: List[str] = field(default_factory=list)

@dataclass
class PhysicalSchema:
    """Physical system metadata: equations, BCs, ICs, units policy, regime tags."""

    physical_system: str = "unknown"
    governing_equations: Dict[str, Any] = field(default_factory=dict)
    ics: Optional[Dict[str, Any]] = None
    bcs: Optional[Dict[str, Any]] = None
    forcings: Optional[Dict[str, Any]] = None
    units_policy: Dict[str, Any] = field(default_factory=dict)
    regime_tags: List[str] = field(default_factory=list)
    validity: Dict[str, Any] = field(default_factory=dict)
    version: str = "upd/v1"

class PhysicalDatasetBuilder:
    """
    MVP-1 builder:
      - time-window sharding (and optional spatial tiles)
      - derived: vorticity/divergence
      - validations: units/ranges/monotonic
      - CLI friendly
    """

    def __init__(self, packs: Optional[Dict[str, List[str]]] = None):
        self.packs = packs or dict(VARIABLE_PACKS)
        self.hub = HubQuery()
        self.spacetime = SpaceTime()
        self.selection = VariableSelection()
        self.schema = PhysicalSchema()

    def login(self, persist: bool = True) -> None:
        """Log in to Earth data hub (earthaccess); persists credentials if requested."""
        earthaccess.login(persist=persist)

    def list_collections(
        self,
        keyword: Optional[str] = None,
        provider: Optional[str] = None,
        short_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search and list available datasets; returns list of metadata dicts."""
        self.login(True)
        kw: Dict[str, Any] = {}
        if keyword: kw["keyword"] = keyword
        if provider: kw["provider"] = provider
        if short_name: kw["short_name"] = short_name
        items = earthaccess.search_datasets(**kw)[:limit]
        out = []
        for it in items:
            d = getattr(it, "data", None)
            if isinstance(d, dict):
                out.append({
                    "title": d.get("title") or d.get("summary") or "untitled",
                    "provider": d.get("provider"),
                    "short_name": d.get("short_name"),
                    "version": d.get("version"),
                    "concept_id": d.get("concept_id") or d.get("id"),
                })
            else:
                out.append({"title": str(it)})
        return out

    def set_dataset(
        self,
        provider: Optional[str] = None,
        short_name: Optional[str] = None,
        keyword: Optional[str] = None,
    ):
        """Set hub query (provider, short_name, keyword); returns self for chaining."""
        self.hub = HubQuery(provider=provider, short_name=short_name, keyword=keyword)
        return self

    def set_spacetime(
        self,
        time_start: Optional[str],
        time_end: Optional[str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        stride: Optional[Dict[str, int]] = None,
        chunking: Optional[Dict[str, int]] = None,
    ):
        """Set temporal range, bbox (W,S,E,N), stride, and chunking; returns self."""
        tr = _temporal(time_start, time_end) if (time_start or time_end) else None
        self.spacetime = SpaceTime(
            time_start=tr[0] if tr else None,
            time_end=tr[1] if tr else None,
            bbox=bbox,
            stride=stride or {},
            chunking=chunking or {},
        )
        return self

    def set_selection(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        packs: Optional[List[str]] = None,
    ):
        """Set variable include/exclude lists and pack names; returns self."""
        self.selection = VariableSelection(include=include or [], exclude=exclude or [], packs=packs or [])
        return self

    def set_schema_from_template(self, template_id: str):
        """Load schema from templates and set it; returns self."""
        tpl = schema_templates().get(template_id)
        if not tpl:
            raise ValueError(f"Template '{template_id}' não encontrado.")
        self.schema = PhysicalSchema(**tpl)
        return self

    def set_schema(self, schema: PhysicalSchema):
        """Set physical schema directly; returns self."""
        self.schema = schema
        return self

    def inspect(
        self,
        prefer: str = "any",
        engine: str = "pydap",
        chunks: Optional[Dict[str, int]] = None,
        max_granules: int = 3,
    ) -> Dict[str, Any]:
        """Open first granule, standardize dims, and return variable/dim metadata and suggested packs."""
        self.login(True)
        granules, notes = self._search_granules(max_granules=max_granules)
        url = _pick_url(granules[0], prefer=prefer)
        ds = xr.open_dataset(url, engine=engine, chunks=(chunks or self.spacetime.chunking or None))
        ds = standardize_dims(ds)
        vars_ = []
        for name, da in ds.data_vars.items():
            vars_.append({
                "name": name,
                "dims": list(da.dims),
                "shape": list(da.shape),
                "dtype": str(da.dtype),
                "units": da.attrs.get("units",""),
                "long_name": (da.attrs.get("long_name") or da.attrs.get("standard_name") or "")[:120],
            })
        vars_.sort(key=lambda x: x["name"])
        return {
            "picked_url": url,
            "dims": {k:int(v) for k,v in ds.sizes.items()},
            "coords": list(ds.coords),
            "variables": vars_,
            "notes": notes,
            "suggested_packs": self._suggest_packs([v["name"] for v in vars_]),
        }

    def build(
        self,
        out_dir: str,
        catalog_path: str,
        shards: Optional[ShardSpec] = None,
        derived: Optional[DerivedSpec] = None,
        validate: Optional[ValidationSpec] = None,
        prefer: str = "any",
        engine: str = "pydap",
        chunks: Optional[Dict[str, int]] = None,
        max_granules: int = 10,
    ) -> Dict[str, Any]:
        """Fetch granules, apply selection/derived/validation, shard, and write UPD zarr + catalog."""
        self.login(True)
        shards = shards or ShardSpec()
        derived = derived or DerivedSpec()
        validate = validate or ValidationSpec(require_units=bool(self.schema.units_policy.get("require_units", True)))

        out = _ensure_dir(out_dir)
        _ensure_dir(Path(catalog_path).parent)

        granules, notes = self._search_granules(max_granules=max_granules)
        written = []
        errors = []

        for g in granules:
            try:
                url = _pick_url(g, prefer=prefer)
                ds = xr.open_dataset(url, engine=engine, chunks=(chunks or self.spacetime.chunking or None))
                ds = standardize_dims(ds)

                # selection
                available = list(ds.data_vars)
                include_eff = list(self.selection.include)
                if self.selection.packs:
                    include_eff += _resolve_packs(self.selection.packs, available, self.packs)
                keep = set(_match(available, include_eff)) if include_eff else set(available)
                drop = set(_match(available, self.selection.exclude)) if self.selection.exclude else set()
                keep = sorted(list(keep - drop))
                if not keep:
                    raise ValueError("Seleção vazia neste granule.")
                ds = ds[keep]

                # stride
                if self.spacetime.stride:
                    isel = {}
                    for dim, st in self.spacetime.stride.items():
                        if dim in ds.dims and isinstance(st, int) and st > 1:
                            isel[dim] = slice(None, None, st)
                    if isel:
                        ds = ds.isel(**isel)

                # derived
                ds = apply_derived(ds, derived)

                # validate
                issues = validate_dataset(ds, validate)
                if issues:
                    raise ValueError("Falhou validação: " + "; ".join(issues[:10]))

                # sharding: time windows + optional tiles
                if "time" not in ds.coords:
                    raise ValueError("Dataset sem coord 'time' (necessário para sharding MVP).")

                for a, b in iter_time_windows(ds["time"].values, shards.time_window):
                    ds_t = subset_time(ds, a, b)
                    if ds_t.sizes.get("time", 0) == 0:
                        continue

                    # optional spatial tiles
                    if shards.tile_deg and ("lat" in ds_t.coords) and ("lon" in ds_t.coords):
                        dlat, dlon = shards.tile_deg
                        for lat0, lat1, lon0, lon1 in iter_tiles(ds_t["lat"].values, ds_t["lon"].values, dlat, dlon):
                            ds_s = subset_tile(ds_t, lat0, lat1, lon0, lon1)
                            if ds_s.sizes.get("lat", 0) == 0 or ds_s.sizes.get("lon", 0) == 0:
                                continue
                            tags = regime_tags_for(ds_s) if shards.add_regime_tags else []
                            written.append(self._write_upd(ds_s, out, catalog_path, url, notes, a, b, tags, tile=(lat0,lat1,lon0,lon1)))
                    else:
                        tags = regime_tags_for(ds_t) if shards.add_regime_tags else []
                        written.append(self._write_upd(ds_t, out, catalog_path, url, notes, a, b, tags, tile=None))

            except Exception as e:
                errors.append({"error": str(e), "granule": str(getattr(g, "data", g))})

        return {
            "written_count": len(written),
            "errors_count": len(errors),
            "written": written,
            "errors": errors[:20],
            "out_dir": str(out),
            "catalog_path": str(catalog_path),
            "notes": notes,
        }

    # -------- internals --------
    def _search_granules(self, max_granules: int) -> Tuple[List[Any], Dict[str, Any]]:
        """Search granules via earthaccess or CMR; returns (granules, notes)."""
        temporal = _temporal(self.spacetime.time_start, self.spacetime.time_end) if (self.spacetime.time_start and self.spacetime.time_end) else None
        bbox = self.spacetime.bbox

        # Try earthaccess first
        kw: Dict[str, Any] = {}
        if self.hub.provider: kw["provider"] = self.hub.provider
        if self.hub.short_name: kw["short_name"] = self.hub.short_name
        if self.hub.keyword and not self.hub.short_name: kw["keyword"] = self.hub.keyword
        if temporal: kw["temporal"] = temporal

        bbox_error = None
        if bbox:
            try:
                kw_bbox = dict(kw); kw_bbox["bounding_box"] = bbox
                granules = earthaccess.search_data(**kw_bbox)
                return granules[:max_granules], {"used": "earthaccess+bbox"}
            except Exception as e:
                bbox_error = str(e)

        try:
            granules = earthaccess.search_data(**kw)
            return granules[:max_granules], {"used": "earthaccess", "bbox_error": bbox_error}
        except Exception as e:
            ea_err = str(e)

        # Fallback python-cmr
        q = GranuleQuery()
        if self.hub.provider: q.provider(self.hub.provider)
        if self.hub.short_name: q.short_name(self.hub.short_name)
        if temporal: q.temporal(temporal[0], temporal[1])
        if bbox:
            w,s,e2,n = bbox
            q.bounding_box(w,s,e2,n)
        items = q.get()
        return items[:max_granules], {"used": "python-cmr", "earthaccess_error": ea_err, "bbox_error": bbox_error}

    def _write_upd(
        self,
        ds: xr.Dataset,
        out_dir: Path,
        catalog_path: str,
        url: str,
        notes: Dict[str, Any],
        t0,
        t1,
        tags: List[str],
        tile: Optional[Tuple[float, float, float, float]],
    ):
        """Write dataset to zarr + JSON meta and append to parquet catalog."""
        # source id includes shard interval + tile (so different uid per shard)
        source_id = f"{self.hub.provider}|{self.hub.short_name or self.hub.keyword}|{url}|{str(t0)}|{str(t1)}|{tile or ''}"
        uid_ = _uid(source_id)

        zarr_path = out_dir / f"{uid_}.zarr"
        meta_path = out_dir / f"{uid_}.json"

        ds.to_zarr(zarr_path, mode="w")

        meta = {
            "uid": uid_,
            "source_id": source_id,
            "fields": list(ds.data_vars),
            "coords": list(ds.coords),
            "dims": {k:int(v) for k,v in ds.sizes.items()},
            "hub_query": asdict(self.hub),
            "spacetime": asdict(self.spacetime),
            "selection": asdict(self.selection),
            "schema": asdict(self.schema),
            "shard": {"time_window": [str(t0), str(t1)], "tile": tile, "regime_tags": tags},
            "notes": notes,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # catalog append
        row = {
            "uid": uid_,
            "zarr_path": str(zarr_path),
            "meta_path": str(meta_path),
            "provider": self.hub.provider,
            "short_name": self.hub.short_name,
            "url": url,
            "time_start": str(t0),
            "time_end": str(t1),
            "tile_json": json.dumps(tile, ensure_ascii=False) if tile else None,
            "regime_tags_json": json.dumps(tags, ensure_ascii=False),
            "n_vars": len(ds.data_vars),
            "dims_json": json.dumps(meta["dims"], ensure_ascii=False),
            "fields_json": json.dumps(meta["fields"], ensure_ascii=False),
        }

        cat_p = Path(catalog_path)
        if cat_p.exists():
            df_old = pd.read_parquet(cat_p)
            df = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True).drop_duplicates(subset=["uid"], keep="last")
        else:
            df = pd.DataFrame([row])
        df.to_parquet(cat_p, index=False)

        return {"uid": uid_, "zarr_path": str(zarr_path), "meta_path": str(meta_path), "regime_tags": tags, "tile": tile}

    def _suggest_packs(self, available_vars: List[str], top_k: int = 5) -> List[str]:
        """Suggest pack names by overlap with available variables."""
        av = set(available_vars)
        scored = []
        for name, vars_ in self.packs.items():
            overlap = sum(1 for v in vars_ if v in av)
            if overlap > 0:
                scored.append((overlap, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[:top_k]]
