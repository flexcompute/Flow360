import flow360 as fl
from flow360 import units as u


def createBaseParams_airplane():
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
            boundaries={},
        )
    return params
