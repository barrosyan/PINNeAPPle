"""Runs INSIDE Abaqus's own Python interpreter (``abaqus python
_abaqus_odb_export_script.py '<json args>'``) -- never imported by regular
PINNeAPPle code, which is why this is a leading-underscore module launched
as a subprocess, not a normal importable one.

Uses only the standard, documented ``odbAccess``/``abaqusConstants`` API
(Abaqus Scripting Reference Guide) and the Python standard library --
deliberately no numpy/torch/PINNeAPPle imports, since Abaqus's bundled
Python environment is not guaranteed to have any third-party package this
repository depends on. Output is plain JSON (a format with zero
dependencies on either side of the subprocess boundary); the calling
process (``abaqus_reader.export_odb_fields``, running in a normal Python
environment with numpy available) converts that JSON to the final
``.npz``.
"""
from __future__ import print_function

import json
import sys


def main():
    args = json.loads(sys.argv[1])
    odb_path = args["odb_path"]
    out_npz_path = args["out_npz_path"]
    step_name = args.get("step_name")
    frame_index = args.get("frame_index", -1)
    field_outputs = args.get("field_outputs", ["U"])

    from odbAccess import openOdb
    from abaqusConstants import NODAL

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        instances = odb.rootAssembly.instances
        if len(instances) == 0:
            raise RuntimeError("odb has no rootAssembly instances")
        # Concatenate nodes/fields across every instance, keyed by a
        # (instance_name, node_label) pair so labels that repeat across
        # instances (legal in Abaqus) don't collide.
        node_keys = []
        coords = []
        for inst_name in instances.keys():
            inst = instances[inst_name]
            for node in inst.nodes:
                node_keys.append((inst_name, node.label))
                c = list(node.coordinates)
                while len(c) < 3:
                    c.append(0.0)
                coords.append(c)
        key_index = {k: i for i, k in enumerate(node_keys)}

        steps = odb.steps
        if step_name is None:
            step_name = list(steps.keys())[-1]
        step = steps[step_name]
        frame = step.frames[frame_index]

        fields_out = {}
        for field_name in field_outputs:
            fo = frame.fieldOutputs[field_name]
            fo_nodal = fo.getSubset(position=NODAL)
            n_components = None
            values_by_index = {}
            for v in fo_nodal.values:
                inst_name = v.instance.name if v.instance is not None else list(instances.keys())[0]
                key = (inst_name, v.nodeLabel)
                if key not in key_index:
                    continue
                data = list(v.data) if hasattr(v.data, "__len__") else [v.data]
                n_components = len(data)
                values_by_index[key_index[key]] = data
            filled = [[0.0] * (n_components or 1) for _ in range(len(node_keys))]
            for idx, data in values_by_index.items():
                filled[idx] = data
            fields_out[field_name] = filled

        payload = {"coords": coords, "fields": fields_out, "step": step_name, "frame_index": frame_index}
        with open(out_npz_path + ".json", "w") as f:
            json.dump(payload, f)
    finally:
        odb.close()


if __name__ == "__main__":
    main()
