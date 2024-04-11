import os

import flow360 as fl
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams

os.environ["FLOW360_BETA_FEATURES"] = "1"

surface_mesh = fl.SurfaceMesh.from_file(
    surface_mesh_file="airplaneGeometry.stl", name="airplane-surface-mesh-beta"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)

params = fl.VolumeMeshingParams(
    type="v2",
    volume=Volume(
        first_layer_thicknes=0.001,
        growth_rate=1.1,
        num_boundary_layers=2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id, name="airplane-volume-mesh-beta", params=params
)
volume_mesh = volume_mesh.submit()
