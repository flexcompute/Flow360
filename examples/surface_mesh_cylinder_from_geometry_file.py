import flow360 as fl
from flow360.examples import CylinderGeometry

fl.Env.dev.active()

params = fl.SurfaceMeshingParams(CylinderGeometry.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    CylinderGeometry.geometry, params=params, name="cylinder-geometry-new-python-client"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
