import flow360 as fl

fl.Env.preprod.active()

project = fl.Project.from_file("wing_tetra.1.lb8.ugrid", name="UDD om6wing from python")
#project = fl.Project.from_cloud(project_id="prj-07341264-d803-4b3f-a4a6-1f44c8e61a13")

volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.4760179762198, 0.801672958512342, 1.4760179762198],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=14.6e6,
            mach=0.84,
            project_length_unit=1 * fl.u.m,
            temperature=297.78,
            alpha=3.06 * fl.u.deg,
            beta=0 * fl.u.deg,
        ),
        models=[
            fl.Wall(surfaces=[volume_mesh["1"]]),
            fl.SlipWall(surfaces=[volume_mesh["2"]]),
            fl.Freestream(surfaces=[volume_mesh["3"]]),
        ],
        time_stepping=fl.Steady(
            max_steps=2000,
        ),
        user_defined_dynamics=[
            fl.UserDefinedDynamic(
                name="alphaController",
                input_vars=["CL"],
                constants={"CLTarget": 0.4, "Kp": 0.2, "Ki": 0.002},
                output_vars={"alphaAngle": "if (pseudoStep > 500) state[0]; else alphaAngle;"},
                state_vars_initial_value=["alphaAngle", "0.0"],
                update_law=[
                    "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
                    "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
                ],
                input_boundary_patches=[volume_mesh["1"]],
            )
        ],
        outputs=[
            fl.VolumeOutput(
                output_fields=[
                    "vorticity",
                    "Cp",
                    "mut",
                    "qcriterion",
                    "Mach",
                ]
            ),
            fl.SurfaceOutput(
                surfaces=volume_mesh["1"],
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                    "yPlus",
                ],
            ),
        ],
    )

project.run_case(params=params, name="UDD om6wing from python")