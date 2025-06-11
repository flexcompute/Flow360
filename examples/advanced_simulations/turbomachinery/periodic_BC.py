import flow360 as fl
from flow360.examples import TutorialPeriodicBC

TutorialPeriodicBC.get_files()
project = fl.Project.from_volume_mesh(
    TutorialPeriodicBC.mesh_filename,
    name="Tutorial Periodic Boundary Condition from Python",
)
volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    slice_inlet = fl.Slice(
        name="Inlet",
        normal=[1, 0, 0],
        origin=[-179, 0, 0] * fl.u.m,
    )
    slice_outlet = fl.Slice(
        name="Outlet",
        normal=[1, 0, 0],
        origin=[539, 0, 0] * fl.u.m,
    )
    slice_trailing_edge = fl.Slice(
        name="TrailingEdge",
        normal=[1, 0, 0],
        origin=[183, 0, 0] * fl.u.m,
    )
    slice_wake = fl.Slice(
        name="Wake",
        normal=[1, 0, 0],
        origin=[294.65, 0, 0] * fl.u.m,
    )
    operating_condition = fl.AerospaceCondition.from_mach_reynolds(
        mach=0.13989,
        reynolds_mesh_unit=3200,
        project_length_unit=1 * fl.u.m,
        temperature=298.25 * fl.u.K,
    )
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0, 0, 0], moment_length=[1, 1, 1], area=209701.3096271187
        ),
        operating_condition=operating_condition,
        time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(max_iterations=20),
                    order_of_accuracy=2,
                    kappa_MUSCL=0.33,
                    update_jacobian_frequency=1,
                    equation_evaluation_frequency=1,
                    numerical_dissipation_factor=1,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    linear_solver=fl.LinearSolver(max_iterations=20),
                    update_jacobian_frequency=1,
                    equation_evaluation_frequency=1,
                    order_of_accuracy=2,
                ),
            ),
            fl.Wall(
                surfaces=[
                    volume_mesh["fluid/vane_*"],  # fluid/vane_ss and vane_ps
                    volume_mesh["fluid/bladeFillet_*"],  # fluid/bladeFillet_ss and bladeFillet_ps
                    volume_mesh["fluid/shroud"],
                    volume_mesh["fluid/hub"],
                ]
            ),
            fl.Freestream(
                surfaces=[
                    volume_mesh["fluid/inlet"],
                ]
            ),
            fl.Outflow(
                surfaces=[
                    volume_mesh["fluid/outlet"],
                ],
                spec=fl.Pressure(1.0032 * operating_condition.thermal_state.pressure),
            ),
            fl.SlipWall(
                surfaces=[
                    volume_mesh["fluid/bottomFront"],
                    volume_mesh["fluid/topFront"],
                ]
            ),
            fl.Periodic(
                surface_pairs=[(volume_mesh["fluid/periodic-1"], volume_mesh["fluid/periodic-2"])],
                spec=fl.Rotational(axis_of_rotation=(1, 0, 0)),
            ),
        ],
        outputs=[
            fl.VolumeOutput(
                output_format="tecplot",
                output_fields=[
                    "primitiveVars",
                    "vorticity",
                    "Cp",
                    "Mach",
                    "qcriterion",
                    "mut",
                    "nuHat",
                    "mutRatio",
                    "gradW",
                    "T",
                    "residualNavierStokes",
                ],
            ),
            fl.SurfaceOutput(
                surfaces=volume_mesh["*"],
                output_format="both",
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                    "yPlus",
                    "nodeForcesPerUnitArea",
                ],
            ),
            fl.SliceOutput(
                slices=[slice_inlet, slice_outlet, slice_trailing_edge, slice_wake],
                output_format="both",
                output_fields=["Cp", "primitiveVars", "T", "Mach", "gradW"],
            ),
        ],
    )


project.run_case(params=params, name="Tutorial Periodic Boundary Condition from Python")
