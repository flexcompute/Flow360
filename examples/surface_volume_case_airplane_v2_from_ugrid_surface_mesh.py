import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl

fl.Env.preprod.active()
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Airplane

surface_mesh_file = "./data/airplane_simple/airplane_simple.lb8.ugrid"

surface_mesh = fl.SurfaceMesh.from_file(surface_mesh_file, name="airplane-surface-mesh-ugrid")
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)

params = fl.VolumeMeshingParams(
    version="v2",
    volume=Volume(
        first_layer_thickness=0.001,
        growth_rate=1.1,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="airplane-volume-mesh-from-ugrid",
    params=params,
    solver_version="mesher-24.2.2",
)
volume_mesh = volume_mesh.submit()

# case
params = fl.Flow360Params(Airplane.case_json)
params.boundaries = {
    "farfield": fl.FreestreamBoundary(),
    "fuselage": fl.NoSlipWall(),
    "leftWing": fl.NoSlipWall(),
    "rightWing": fl.NoSlipWall(),
}
case_draft = volume_mesh.create_case("airplane-case-from-ugrid-surface-mesh-v2", params)
case = case_draft.submit()
