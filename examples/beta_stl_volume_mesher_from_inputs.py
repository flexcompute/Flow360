import flow360 as fl
from flow360.component.meshing.params import VolumeMeshingParams, Volume,\
    Farfield

import os
os.environ["FLOW360_BETA_FEATURES"] = '1'

surface_mesh = fl.SurfaceMesh.create(
    geometry_file = "airplaneGeometry.stl", params=None, name="airplane-surface-mesh-beta"
)
surface_mesh = surface_mesh.submit()
print(surface_mesh)
print(surface_mesh.params)

params = fl.VolumeMeshingParams(
    type = "beta",
    volume = Volume(
        first_layer_thicknes=0.001,
        growth_rate=1.1,
        num_boundary_layers=2,
    ),
    farfield = Farfield(type="auto")
)

volume_mesh = surface_mesh.create_volume_mesh("airplane-volume-mesh-beta", params=params)
volume_mesh = volume_mesh.submit()
