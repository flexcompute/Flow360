import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl
from flow360.examples import Airplane

fl.Env.preprod.active()

surface_mesh_stl = "../tests/data/surface_mesh/airplaneGeometry.stl"

surface_mesh = fl.SurfaceMesh.from_file(surface_mesh_stl, name="airplane-surface-mesh-stl")

surface_mesh = surface_mesh.submit(force_submit=True)

print(surface_mesh)
print(surface_mesh.params)
