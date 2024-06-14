import os

import flow360 as fl

fl.Env.preprod.active()

from flow360.component.geometry import Geometry
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Cylinder3D

# geometry
geometry_draft = Geometry.from_file(
    Cylinder3D.geometry, name="testing-cylinder3d-3rd-party-geometry"
)
geometry = geometry_draft.submit()
print(geometry)

# surface mesh
params = fl.SurfaceMeshingParams(max_edge_length=0.5)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_id=geometry.id,
    params=params,
    name="cylinder3d-surface-mesh-from-3rd-party-geometry-v1",
)
surface_mesh = surface_mesh_draft.submit()

print(surface_mesh)

# volume mesh
params = fl.VolumeMeshingParams(
    volume=Volume(
        first_layer_thickness=1e-4,
        growth_rate=1.2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="cylinder3d-volume-mesh-from-3rd-party-geometry-id-v1",
    params=params,
)
volume_mesh = volume_mesh_draft.submit()
print(volume_mesh)

# case
params = fl.Flow360Params(Cylinder3D.case_json)
params.boundaries = {
    "fluid/farfield": fl.FreestreamBoundary(),
    "fluid/unspecified": fl.NoSlipWall(),
}
case_draft = volume_mesh.create_case(
    "cylinder3d-case-from-egads-3rd-party-geometry-id-v1", params, solver_version="mesher-24.2.2"
)
case = case_draft.submit()
