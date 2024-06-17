import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl

fl.Env.preprod.active()

from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Airplane

# surface mesh
params = fl.SurfaceMeshingParams(version="v2", max_edge_length=0.16)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_file=Airplane.geometry,
    params=params,
    name="airplane-surface-mesh-from-geometry-v2",
    solver_version="mesher-24.2.1",
)
surface_mesh = surface_mesh_draft.submit()

print(surface_mesh)

# volume mesh
params = fl.VolumeMeshingParams(
    version="v2",
    volume=Volume(
        first_layer_thickness=1e-5,
        growth_rate=1.2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="airplane-volume-mesh-from-geometry-v2",
    params=params,
    solver_version="mesher-24.2.1",
)
volume_mesh = volume_mesh_draft.submit()
print(volume_mesh)

# case
params = fl.Flow360Params(Airplane.case_json)
params.boundaries = {
    "farfield": fl.FreestreamBoundary(),
    "fuselage": fl.NoSlipWall(),
    "leftWing": fl.NoSlipWall(),
    "rightWing": fl.NoSlipWall(),
}
case_draft = volume_mesh.create_case("airplane-case-from-csm-geometry-v2", params)
case = case_draft.submit()