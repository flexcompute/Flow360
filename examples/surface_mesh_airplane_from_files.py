from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.v1.meshing.params import SurfaceMeshingParams
from flow360.examples import Airplane

params = SurfaceMeshingParams(Airplane.surface_json)
surface_mesh = SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
