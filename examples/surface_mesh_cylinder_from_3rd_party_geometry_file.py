import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder

fl.Env.preprod.active()

geometry = Geometry.from_file(
    Cylinder.geometry,
    name="testing-cylinder-geometry",
)
geometry = geometry.submit()

params = fl.SurfaceMeshingParams(Cylinder.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    geometry_id=geometry.id, params=params, name="cylinder-surface-mesh"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
