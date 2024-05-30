import os

import flow360 as fl

fl.Env.preprod.active()

from simulation_examples.airplane_case import createBaseParams_airplane

from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams

# surface mesh
params = fl.SurfaceMeshingParams(max_edge_length=0.16)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_file="data/airplane_geometry.egads",
    params=params,
    name="airplane-surface-mesh-from-egads-geometry-v1",
    solver_version="mesher-24.2.1",
)
surface_mesh = surface_mesh_draft.submit()

print(surface_mesh)
print(surface_mesh.params)

# volume mesh
params = fl.VolumeMeshingParams(
    volume=Volume(
        first_layer_thickness=1e-5,
        growth_rate=1.2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="airplane-volume-mesh-from-egads-geometry-v1",
    params=params,
    solver_version="mesher-24.2.1",
)
volume_mesh = volume_mesh_draft.submit()

# case
params = createBaseParams_airplane()
params.boundaries = {
    "fluid/farfield": fl.FreestreamBoundary(),
    "fluid/fuselage": fl.NoSlipWall(),
    "fluid/leftWing": fl.NoSlipWall(),
    "fluid/rightWing": fl.NoSlipWall(),
}
case_draft = volume_mesh.create_case("airplane-case-from-egads-geometry-v1", params)
case = case_draft.submit()
