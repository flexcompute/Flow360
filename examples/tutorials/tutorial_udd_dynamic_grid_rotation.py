import flow360 as fl
from flow360.examples import TutorialUDDDynamicGridRotation

TutorialUDDDynamicGridRotation.get_files()

project = fl.Project.from_file(
    files=fl.VolumeMeshFile(TutorialUDDDynamicGridRotation.mesh_filename),
    name="Tutorial UDD dynamic grid rotation from Python",
)

volume_mesh = project.volume_mesh


with fl.SI_unit_system:
    rotation_zone = volume_mesh["plateBlock"]
    rotation_zone.center = [0, 0, 0]
    rotation_zone.axis = [0, 1, 0]
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=0.5325,
            moment_center=[0, 0, 0],
            moment_length=[1, 1, 1],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=31794.3326488706,
            mach=0.2,
            project_length_unit=1 * fl.u.m,
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    relative_tolerance=1e-3,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    kappa_MUSCL=0.33,
                    update_jacobian_frequency=1,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    relative_tolerance=1e-2,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    rotation_correction=True,
                    update_jacobian_frequency=1,
                    equation_evaluation_frequency=1,
                ),
            ),
            fl.Wall(name="NoSlipWall", surfaces=[volume_mesh["plateBlock/noSlipWall"]]),
            fl.SlipWall(name="SlipWall", surfaces=[volume_mesh["*/slipWall"]]),
            fl.Freestream(name="Freestream", surfaces=[volume_mesh["farFieldBlock/farField"]]),
            fl.Rotation(
                volumes=rotation_zone,
                spec=fl.FromUserDefinedDynamics(),
            ),
        ],
        time_stepping=fl.Unsteady(
            steps=500,
            max_pseudo_steps=100,
            step_size=0.0014693176825506479,
            CFL=fl.RampCFL(initial=1, final=50000, ramp_steps=5),
        ),
        user_defined_dynamics=[
            fl.UserDefinedDynamic(
                name="dynamicTheta",
                input_vars=["momentY"],
                constants={
                    "I": 0.443768309310345,
                    "zeta": 0.014,
                    "K": 0.023216703348186308,
                    "omegaN": 0.22872946666666666,
                    "theta0": 0.17453292519943295,
                },
                output_vars={"omegaDot": "state[0];", "omega": "state[1];", "theta": "state[2];"},
                state_vars_initial_value=["-1.82621384e-02", "0.0", "0.5235987755982988"],
                update_law=[
                    "if (pseudoStep == 0) (momentY - K * ( state[2] - theta0 ) - 2 * zeta * omegaN * I *state[1] ) / I; else state[0];",
                    "if (pseudoStep == 0) state[1] + state[0] * timeStepSize; else state[1];",
                    "if (pseudoStep == 0) state[2] + state[1] * timeStepSize; else state[2];",
                ],
                input_boundary_patches=[volume_mesh["plateBlock/noSlipWall"]],
                output_target=volume_mesh["plateBlock"],
            )
        ],
        outputs=[
            fl.VolumeOutput(output_fields=["primitiveVars"]),
            fl.SurfaceOutput(
                surfaces=volume_mesh["plateBlock/noSlipWall"],
                output_fields=["Cp"],
            ),
        ],
    )


project.run_case(params=params, name="Case of tutorial UDD dynamic grid rotation from Python")
