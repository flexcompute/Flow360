import flow360 as fl
from flow360.examples import TutorialCHTSolver
from flow360 import log, u
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    BottomCamera,
    Chart2D,
    Chart3D,
    FrontCamera,
    FrontLeftBottomCamera,
    FrontLeftTopCamera,
    Inputs,
    LeftCamera,
    RearCamera,
    RearLeftTopCamera,
    RearRightBottomCamera,
    Settings,
    Summary,
    Table,
    TopCamera,
)

TutorialCHTSolver.get_files()
project = fl.Project.from_file(
    files=fl.VolumeMeshFile(TutorialCHTSolver.mesh_filename), name="Tutorial CHT Solver from Python"
)
volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0, 0, 0] * fl.u.m,
            moment_length=[1, 1, 1] * fl.u.m,
            area=1 * fl.u.m**2,
        ),
        operating_condition=fl.AerospaceCondition.from_mach(mach=0.1),
        time_stepping=fl.Steady(
            max_steps=10000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=1000)
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    order_of_accuracy=2,
                    kappa_MUSCL=-1,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    equation_evaluation_frequency=4,
                    order_of_accuracy=2,
                ),
            ),
            fl.Solid(
                entities=volume_mesh["solid"],
                heat_equation_solver=fl.HeatEquationSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(
                        max_iterations=25,
                        absolute_tolerance=1e-12,
                    ),
                    equation_evaluation_frequency=10,
                ),
                material=fl.SolidMaterial(
                    name="copper",
                    thermal_conductivity=398 * fl.u.W / (fl.u.m * fl.u.K),
                ),
                volumetric_heat_source=5e3 * fl.u.W / (0.01257 * fl.u.m**3),
            ),
            fl.Wall(
                surfaces=volume_mesh["fluid/centerbody"],
            ),
            fl.Freestream(
                surfaces=volume_mesh["fluid/farfield"],
            ),
            fl.Wall(
                surfaces=volume_mesh["solid/adiabatic"],
                heat_spec=fl.HeatFlux(0 * fl.u.W / fl.u.m**2),
            ),
        ],
        outputs=[
            fl.VolumeOutput(
                name="fl.VolumeOutput",
                output_format="both",
                output_fields=[
                    "primitiveVars",
                    "T",
                    "Cp",
                    "Mach",
                ],
            ),
            fl.SurfaceOutput(
                name="fl.SurfaceOutput",
                surfaces=volume_mesh["*"],
                output_format="both",
                output_fields=["primitiveVars", "T", "Cp", "Cf", "CfVec"],
            ),
        ],
    )

project.run_case(params=params, name="Tutorial CHT Solver from Python")






top_camera = TopCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")
top_camera_slice = TopCamera(pan_target=(2.5, 0, 0), dimension=8, dimension_dir="width")
side_camera = LeftCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")
side_camera_slice = LeftCamera(pan_target=(2.5, 0, 1.5), dimension=8, dimension_dir="width")
rear_camera = RearCamera(dimension=2.5, dimension_dir="width")
front_camera = FrontCamera(dimension=2.5, dimension_dir="width")
bottom_camera = BottomCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")


x_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocity",
        limits=(250 * u.K, 500 * u.K),
        camera=side_camera_slice,
        fig_name=f"slice_y_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]

y_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocity",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=side_camera_slice,
        fig_name=f"slice_y_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]




report = ReportTemplate(
    title="CHT results reporting",
    items=[
        Summary(),
        Inputs(),
        statistical_table,
        Chart2D(
            x="x_slicing_force_distribution/X",
            y="x_slicing_force_distribution/totalCumulative_CD_Curve",
            fig_name="totalCumulative_CD_Curve",
            background="geometry",
            exclude=exclude,
        ),
        Chart2D(
            x="surface_forces/pseudo_step",
            y="surface_forces/totalCD",
            section_title="Drag Coefficient",
            fig_name="cd_fig",
            exclude=exclude,
            focus_x=(1 / 3, 1),
        ),
        *geometry_screenshots,
        *cpt_screenshots,
        *y_slices_screenshots,
        # *y_slices_lic_screenshots,
        *z_slices_screenshots,
        *y_plus_screenshots,
        *cp_screenshots,
        *cpx_screenshots,
        *wall_shear_screenshots,
    ],
    settings=Settings(dpi=150),
)

report = report.create_in_cloud(
    f"CHT-slices-using-groups-Cpt, Cpx, wallShear, dpi=default",
    cases,
    solver_version=SOLVER_VERSION,
)