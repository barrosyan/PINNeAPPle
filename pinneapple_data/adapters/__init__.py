try:
    from .geom_adapter import (
        GeometryInput,
        GeometryLoadOptions,
        load_geometry_asset,
        attach_geometry,
        stl_to_upd,
    )
except Exception:
    pass

try:
    from .upd_adapter import (
        UPDInput,
        load_upd_item,
        upd_to_physical_sample,
        attach_upd_state,
    )
except Exception:
    pass

from .pinn_batch_builders import (
    PINNBatch,
    build_from_bundle,
    build_from_solver,
    build_from_real_data,
)

from .splash_archive_adapter import (
    SplashMesh,
    FieldSnapshot,
    SplashCase,
    open_case,
    load_mesh,
    save_mesh_cache,
    load_mesh_cache,
    load_dense_volumes,
)

__all__ = [
    "GeometryInput",
    "GeometryLoadOptions",
    "load_geometry_asset",
    "attach_geometry",
    "stl_to_upd",
    "UPDInput",
    "load_upd_item",
    "upd_to_physical_sample",
    "attach_upd_state",
    "PINNBatch",
    "build_from_bundle",
    "build_from_solver",
    "build_from_real_data",
    "SplashMesh",
    "FieldSnapshot",
    "SplashCase",
    "open_case",
    "load_mesh",
    "save_mesh_cache",
    "load_mesh_cache",
    "load_dense_volumes",
]
