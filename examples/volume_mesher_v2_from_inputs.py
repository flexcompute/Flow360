import os

import flow360 as fl
os.environ["FLOW360_BETA_FEATURES"] = "1"

fl.Env.dev.active()
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams

surface_mesh_stl = "../tests/data/surface_mesh/airplaneGeometry.stl"

surface_mesh = fl.SurfaceMesh.from_file(surface_mesh_stl, name="airplane-surface-mesh-stl")
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)

params = fl.VolumeMeshingParams(
    version="v2",
    volume=Volume(
        first_layer_thickness=0.001,
        growth_rate=1.1,
        num_boundary_layers=2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id, name="airplane-volume-mesh-from-stl", params=params
)
volume_mesh = volume_mesh.submit(force_submit=True)
