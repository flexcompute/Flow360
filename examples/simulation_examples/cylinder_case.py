import flow360 as fl
from flow360 import units as u

def createBaseParams_cylinder():
    mesh_unit = 1 * u.m
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=340,
                moment_length=(1, 1, 1),
                moment_center=(0, 0, 0),
                mesh_unit=mesh_unit,
            ),
            volume_output=fl.VolumeOutput(
                output_format="tecplot",
                output_fields=["primitiveVars", "qcriterion"],
            ),
            surface_output=fl.SurfaceOutput(
                output_format="tecplot",
                output_fields=[
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
            ),
            turbulence_model_solver=fl.SpalartAllmaras(
                absolute_tolerance=1e-8,
                relative_tolerance=1e-2,
                linear_solver=fl.LinearSolver(max_iterations=35),
            ),
            freestream=fl.FreestreamFromMachReynolds(
                Mach=0.1, Reynolds=5, temperature=288.15, alpha=0, beta=0
            ),
            fluid_properties=fl.air,
            time_stepping=fl.SteadyTimeStepping(
                max_pseudo_steps=2000,
                CFL=fl.AdaptiveCFL(),
            ),
            boundaries={},
        )
    return params
