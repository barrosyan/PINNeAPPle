import trimesh

from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
from pinneaple_geom.ops.repair import repair_mesh
from pinneaple_geom.ops.simplify import simplify_mesh

mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
mesh.vertices[0] = mesh.vertices[1]  # introduce a tiny issue

bridge = TrimeshBridge()
g = bridge.from_trimesh(mesh)

g2 = repair_mesh(g)
g3 = simplify_mesh(g2, target_faces=5000, backend="trimesh")

print("Number of faces:", g3.n_faces)
print("Number of vertices:", g3.n_vertices)
print("Normals:", g3.normals)