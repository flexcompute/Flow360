import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl
from flow360 import units as u

fl.Env.preprod.active()
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Airplane


def createBaseParams():
    mesh_unit = 1 * u.m
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1,
                moment_length=(1, 1, 1),
                moment_center=(0, 0, 0),
                mesh_unit=mesh_unit,
            ),
            volume_output=fl.VolumeOutput(
                output_format="tecplot",
                output_fields=["primitiveVars", "qcriterion"],
            ),
            surface_output=fl.SurfaceOutput(
                output_format="both",
                output_fields=[
                    "nuHat",
                    "Cp",
                    "Cf",
                    "CfVec",
                    "yPlus",
                    "nodeForcesPerUnitArea",
                ],
            ),
            navier_stokes_solver=fl.NavierStokesSolver(
                absolute_tolerance=1e-11,
                relative_tolerance=1e-3,
                linear_solver=fl.LinearSolver(max_iterations=35),
                kappa_MUSCL=0.33,
                order_of_accuracy=2,
                update_jacobian_frequency=4,
                equation_eval_frequency=1,
            ),
            turbulence_model_solver=fl.SpalartAllmaras(
                absolute_tolerance=1e-8,
                relative_tolerance=1e-2,
                linear_solver=fl.LinearSolver(max_iterations=35),
                order_of_accuracy=2,
                update_jacobian_frequency=4,
                equation_eval_frequency=4,
            ),
            freestream=fl.FreestreamFromMach(
                Mach=0.4, mu_ref=1e-6, temperature=288.15, alpha=0, beta=0
            ),
            fluid_properties=fl.air,
            time_stepping=fl.SteadyTimeStepping(
                max_pseudo_steps=4000,
                CFL=fl.AdaptiveCFL(),
            ),
            boundaries={
                "farfield": fl.FreestreamBoundary(),
                "fuselage": fl.NoSlipWall(),
                "leftWing": fl.NoSlipWall(),
                "rightWing": fl.NoSlipWall(),
            },
        )
    return params


# surface mesh
params = fl.SurfaceMeshingParams(version="v2", max_edge_length=0.16)

surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry,
    params=params,
    name="airplane-new-python-client-v2",
    solver_version="mesher-24.2.1"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)

# volume mesh
params = fl.VolumeMeshingParams(
    version="v2",
    volume=Volume(
        first_layer_thickness=1e-5,
        growth_rate=1.2,
    ),
    farfield=Farfield(type="auto"),
)

volume_mesh = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name="airplane-volume-mesh-from-geometry",
    params=params,
    solver_version="mesher-24.2.1"
)
volume_mesh = volume_mesh.submit()

# case
params = createBaseParams()
case_draft = volume_mesh.create_case("airplane-case-from-geometry-v2", params)
case = case_draft.submit()
