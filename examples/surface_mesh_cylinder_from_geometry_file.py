import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import CylinderGeometry

fl.Env.dev.active()

geometry = Geometry.from_file(
    CylinderGeometry.geometry, name="testing-cylinder-geometry", solver_version="release-24.2.2"
)
geometry = geometry.submit()

params = fl.SurfaceMeshingParams(CylinderGeometry.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    geometry_id=geometry.id, params=params, name="cylinder-geometry-new-python-client"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
