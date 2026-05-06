import trimesh
import numpy as np

from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
from pinneaple_geom.sample.points import sample_surface_points

# create trimesh geometry
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

# bridge
bridge = TrimeshBridge()
g = bridge.from_trimesh(mesh)

# sample points
pts, normals, face_id = sample_surface_points(g, n=10_000)

pts = np.asarray(pts)

print("mesh faces:", len(mesh.faces))
print("sample points:", pts.shape, "min/max", pts.min(), pts.max())
