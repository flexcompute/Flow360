import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder3D

fl.Env.preprod.active()

geometry = Geometry.from_file(
    Cylinder3D.geometry,
    name="testing-cylinder3d-geometry",
)
geometry = geometry.submit()

params = fl.SurfaceMeshingParams(Cylinder3D.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    geometry_id=geometry.id, params=params, name="cylinder3d-surface-mesh"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
