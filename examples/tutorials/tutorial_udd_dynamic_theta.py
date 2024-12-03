import flow360 as fl

fl.Env.preprod.active()

project = fl.Project.from_file("XV15_Hover_ascent_coarse_v2.cgns", name="UDD XV15 from Python")

volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    rotation_zone = volume_mesh["innerRotating"]
    rotation_zone.center = [0, 0, 0]
    rotation_zone.axis = [0, 0, -1]
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=45.604,
            moment_center=[0, 0, 0],
            moment_length=[3.81, 3.81, 3.81],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=3.42369e5,
            mach=1.46972e-02,
            project_length_unit=1 * fl.u.m,
            temperature=288.15,
            alpha=-90 * fl.u.deg,
            reference_mach=0.70
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    relative_tolerance=1e-2,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    relative_tolerance=1e-2,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    DDES=True,
                    equation_evaluation_frequency=1,
                )
            ),
            fl.Wall(
                name="NoSlipWall",
                surfaces=[volume_mesh["innerRotating/blade"]]
            ),
            fl.Freestream(
                name="Freestream",
                surfaces=[volume_mesh["farField/farField"]]
            ),
            fl.Rotation(
                volumes=rotation_zone,
                spec=fl.AngularVelocity(600 * fl.u.rpm)
            )
        ],
        time_stepping=fl.Unsteady(
            step_size=2.83500e-01,
            steps=600,
            max_pseudo_steps=35,
        ),
        outputs=[
            fl.VolumeOutput(
                output_fields=["primitiveVars", "T", "Mach", "qcriterion", "VelocityRelative"]
            ),
            fl.SurfaceOutput(
                output_fields=["primitiveVars", "Cp", "Cf", "CfTangent", "CfNormal", "yPlus", "nodeForcesPerUnitArea"]
            )
        ],
        user_defined_dynamics=[
            fl.UserDefinedDynamic(
                name="dynamicTheta",
                input_vars=["momentY"],
                constants={
                    "I": 0.443768309310345,
                    "zeta": 4.0,
                    "K": 0.0161227107,
                    "omegaN": 0.190607889,
                    "theta0": 0.0872664626
                },
                output_vars={
                    "omegaDot": "state[0]",
                    "omega": "state[1]",
                    "theta": "state[2]",
                },
                state_vars_initial_value=[
                    "-1.82621384e-02",
                    "0.0",
                    "1.39626340e-01",
                ],
                update_law=[
                    "if (pseudoStep == 0) (momentY - K * ( state[2] - theta0 ) - 2 * zeta * omegaN * I *state[1] ) / I; else state[0];",
                    "if (pseudoStep == 0) state[1] + state[0] * timeStepSize; else state[1];",
                    "if (pseudoStep == 0) state[2] + state[1] * timeStepSize; else state[2];"
                ],
                input_boundary_patches=["plateBlock/noSlipWall"],
                output_target=rotation_zone
            ),
        ]
    )