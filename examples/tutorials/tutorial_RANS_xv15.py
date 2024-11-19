import math

import flow360 as fl
from flow360.examples import TutorailRANSXv15

fl.Env.preprod.active()

TutorailRANSXv15.get_files()
project = fl.Project.from_file(
    TutorailRANSXv15.mesh_filename,
    name="Tutorial Time-accurate RANS CFD on XV-15 from Python",
)
volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    rotation_cylinder = volume_mesh["innerRotating"]
    rotation_cylinder.center = (0, 0, 0) * fl.u.m
    rotation_cylinder.axis = (0, 0, -1)
    farfield = fl.AutomatedFarfield(name="farfield")
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0, 0, 0),
            moment_length=(3.81, 3.81, 3.81),
            area=45.604,
        ),
        operating_condition=fl.AerospaceCondition.from_mach(
            mach=0.0146972,
            reference_mach=0.7,
            alpha=-90 * fl.u.deg,
            beta=0 * fl.u.deg,
            thermal_state=fl.ThermalState(
                temperature=288.15,
                material=fl.Air(
                    dynamic_viscosity=fl.Sutherland(
                        reference_temperature=288.15,
                        reference_viscosity=4.29279e-8 * fl.u.flow360_viscosity_unit,
                        effective_temperature=110.4,
                    )
                ),
            ),
        ),
        time_stepping=fl.Unsteady(
            max_pseudo_steps=35,
            steps=600,
            step_size=0.5 / 600 * fl.u.s,
            CFL=fl.AdaptiveCFL(convergence_limiting_factor=0.25),
        ),
        outputs=[
            fl.VolumeOutput(
                name="VolumeOutput",
                output_fields=[
                    "primitiveVars",
                    "T",
                    "Cp",
                    "Mach",
                    "qcriterion",
                    "VelocityRelative",
                ],
            ),
            fl.SurfaceOutput(
                name="SurfaceOutput",
                surfaces=volume_mesh["*"],
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                    "yPlus",
                    "nodeForcesPerUnitArea",
                ],
            ),
        ],
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    DDES=True,
                    rotation_correction=True,
                    equation_evaluation_frequency=1,
                ),
            ),
            fl.Rotation(
                volumes=rotation_cylinder,
                spec=fl.AngularVelocity(20 * math.pi * fl.u.rad / fl.u.s),
            ),
            fl.Freestream(surfaces=volume_mesh["farField/farField"], name="Freestream"),
            fl.Wall(surfaces=volume_mesh["innerRotating/blade"], name="NoSlipWall"),
        ],
    )

project.run_case(
    params=params,
    name="Tutorial Time-accurate RANS CFD on XV-15 from Python",
)
