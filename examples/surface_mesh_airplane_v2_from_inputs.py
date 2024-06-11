import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl
from flow360.examples import Airplane

fl.Env.preprod.active()

params = fl.SurfaceMeshingParams(version="v2", max_edge_length=0.16)

surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry,
    params=params,
    name="airplane-new-python-client-v2",
    solver_version="mesher-24.2.1",
)
surface_mesh = surface_mesh.submit(force_submit=True)

print(surface_mesh)
print(surface_mesh.params)
