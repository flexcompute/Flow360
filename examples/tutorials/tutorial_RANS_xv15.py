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
    rotation_zone = volume_mesh["innerRotating"]
    rotation_zone.center = (0, 0, 0) * fl.u.m
    rotation_zone.axis = (0, 0, -1)
    farfield = fl.AutomatedFarfield(name="farfield")
    thermal_state = fl.ThermalState(
        temperature=288.15,
        material=fl.Air(
            dynamic_viscosity=fl.Sutherland(
                reference_temperature=288.15,
                reference_viscosity=4.29279e-8 * fl.u.flow360_viscosity_unit,
                effective_temperature=110.4,
            )
        ),
    )
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0, 0, 0),
            moment_length=(3.81, 3.81, 3.81),
            area=45.604,
        ),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=5.0 * fl.u.m / fl.u.s,
            reference_velocity_magnitude=0.7 * thermal_state.speed_of_sound,
            alpha=-90 * fl.u.deg,
            beta=0 * fl.u.deg,
            thermal_state=thermal_state,
        ),
        time_stepping=fl.Unsteady(
            max_pseudo_steps=35,
            steps=600,
            step_size=0.5 / 600 * fl.u.s,
            CFL=fl.AdaptiveCFL(),
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
                    limit_velocity=True,
                    limit_pressure_density=True,
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
                volumes=rotation_zone,
                spec=fl.AngularVelocity(600 * fl.u.rpm),
            ),
            fl.Freestream(surfaces=volume_mesh["farField/farField"], name="Freestream"),
            fl.Wall(surfaces=volume_mesh["innerRotating/blade"], name="NoSlipWall"),
        ],
    )

project.run_case(
    params=params,
    name="Tutorial Time-accurate RANS CFD on XV-15 from Python",
)
