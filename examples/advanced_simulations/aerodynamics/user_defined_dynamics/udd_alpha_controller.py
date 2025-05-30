import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename, name="Tutorial UDD alpha controller from Python"
)

volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315,
            moment_center=[0, 0, 0],
            moment_length=[1.47601, 0.80167, 1.47601],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=11.72e6,
            mach=0.84,
            project_length_unit=0.80167 * fl.u.m,
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

project.run_case(params=params, name="Case of tutorial UDD alpha controller from Python")
