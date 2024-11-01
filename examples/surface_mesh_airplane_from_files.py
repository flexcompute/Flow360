import flow360.component.v1.modules as fl
from flow360.examples import Airplane

params = fl.SurfaceMeshingParams(Airplane.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
