import flow360 as fl
from flow360.examples import TutorialRANSXv15

TutorialRANSXv15.get_files()
project = fl.Project.from_volume_mesh(
    TutorialRANSXv15.mesh_filename,
    name="Tutorial Time-accurate RANS CFD on XV-15 from Python",
)
volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    rotation_zone = volume_mesh["innerRotating"]
    rotation_zone.center = (0, 0, 0) * fl.u.m
    rotation_zone.axis = (0, 0, -1)
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0, 0, 0),
            moment_length=(3.81, 3.81, 3.81),
            area=45.604,
        ),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=5,
            alpha=-90 * fl.u.deg,
            reference_velocity_magnitude=238.14,
        ),
        time_stepping=fl.Unsteady(
            max_pseudo_steps=35,
            steps=600,
            step_size=0.5 / 600 * fl.u.s,
            CFL=fl.AdaptiveCFL(),
        ),
        outputs=[
            fl.VolumeOutput(
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
                    hybrid_model=fl.DetachedEddySimulation(shielding_function="DDES"),
                    rotation_correction=True,
                    equation_evaluation_frequency=1,
                ),
            ),
            fl.Rotation(
                volumes=rotation_zone,
                spec=fl.AngularVelocity(600 * fl.u.rpm),
            ),
            fl.Freestream(surfaces=volume_mesh["farField/farField"]),
            fl.Wall(surfaces=volume_mesh["innerRotating/blade"]),
        ],
    )

project.run_case(
    params=params,
    name="Tutorial Time-accurate RANS CFD on XV-15 from Python",
)
