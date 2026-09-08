"""Open-format CFD/FEM readers beyond OpenFOAM: CGNS, Exodus II, Fluent
mesh (ASCII, geometry-only), an Abaqus .inp/.odb bridge, and STAR-CCM+
(CGNS export delegation + table/CSV export).

Validation status differs sharply by module -- read each one's own
docstring before trusting its output for anything important:

- ``openfoam`` (sibling package, not here): validated byte-for-byte
  against a real 244 MB LES case.
- ``cgns_reader``, ``exodus_reader``, ``fluent_mesh_reader``: implement
  their documented open-format specs precisely, but were NOT checked
  against a real file from a real writer (none was available) -- treat as
  unverified until exercised against one.
- ``abaqus_reader``'s ``.inp`` mesh parser: an open text format, parsed
  directly, same confidence level as the OpenFOAM/CGNS/Exodus readers.
- ``abaqus_reader``'s ``.odb`` bridge: correctness depends entirely on a
  local, licensed Abaqus installation's own ``odbAccess`` API -- this
  repository does not and cannot implement the (undocumented, proprietary)
  ``.odb`` binary format itself.
- ``starccm_reader``'s CGNS path: delegates entirely to ``cgns_reader``
  (no separate implementation to validate). Its table/CSV parser is a
  real, implemented parser tested against a hand-constructed synthetic
  fixture (no real STAR-CCM+ file was available); its ``.sim``/``.ccm``
  native binary format is explicitly NOT supported -- see the module
  docstring for why.
"""
from .cgns_reader import read_cgns_mesh_and_fields, cgns_to_upd
from .exodus_reader import read_exodus, exodus_to_upd
from .fluent_mesh_reader import read_fluent_mesh_nodes, fluent_mesh_to_upd
from .abaqus_reader import (
    read_abaqus_inp_mesh, abaqus_inp_to_upd,
    export_odb_fields, load_exported_odb_npz,
)
from .starccm_reader import (
    read_starccm_cgns, starccm_cgns_to_upd,
    read_starccm_table, starccm_table_to_upd,
)

__all__ = [
    "read_cgns_mesh_and_fields", "cgns_to_upd",
    "read_exodus", "exodus_to_upd",
    "read_fluent_mesh_nodes", "fluent_mesh_to_upd",
    "read_abaqus_inp_mesh", "abaqus_inp_to_upd",
    "export_odb_fields", "load_exported_odb_npz",
    "read_starccm_cgns", "starccm_cgns_to_upd",
    "read_starccm_table", "starccm_table_to_upd",
]
