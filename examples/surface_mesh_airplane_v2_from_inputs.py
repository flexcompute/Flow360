import flow360 as fl
from flow360.examples import Airplane

os.environ["FLOW360_BETA_FEATURES"] = "1"

params = fl.SurfaceMeshingParams(version="v2", max_edge_length=0.16)

surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client-v2"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)