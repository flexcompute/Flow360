import os

import flow360 as fl

fl.Env.preprod.active()

from flow360.component.geometry import Geometry
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Cylinder3D

geometry = Geometry.from_file(Cylinder3D.geometry, name="cylinder3d-geometry")
geometry = geometry.submit()

print(geometry)

# surface mesh
params = fl.SurfaceMeshingParams(max_edge_length=10)

surface_mesh = fl.SurfaceMesh.create(
    geometry_id=geometry.id,
    params=params,
    name="cylinder3d-surface-mesh-from-geometry",
)
surface_mesh = surface_mesh.submit(force_submit=True)

print(surface_mesh)
print(surface_mesh.params)

# volume mesh
params = fl.VolumeMeshingParams(
    volume=Volume(
        first_layer_thickness=1e-3,
        growth_rate=1.2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="cylinder3d-volume-mesh-from-geometry",
    params=params,
)
volume_mesh = volume_mesh.submit(force_submit=True)
