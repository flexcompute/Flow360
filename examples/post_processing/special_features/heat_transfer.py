import flow360 as fl
from flow360 import u
from flow360.examples import TutorialCHTSolver
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    Camera,
    Chart3D,
    FrontCamera,
    Inputs,
    LeftCamera,
    Settings,
    Summary,
)
from flow360.version import __solver_version__

TutorialCHTSolver.get_files()

project = fl.Project.from_volume_mesh(
    TutorialCHTSolver.mesh_filename, name="CHT results from Python"
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
                output_format="both",
                output_fields=[
                    "primitiveVars",
                    "T",
                    "Cp",
                    "Mach",
                ],
            ),
            fl.SurfaceOutput(
                surfaces=volume_mesh["*"],
                output_format="both",
                output_fields=["primitiveVars", "T", "Cp", "Cf", "CfVec"],
            ),
            fl.SliceOutput(
                entities=[
                    fl.Slice(
                        name="slice_x",
                        normal=(1, 0, 0),
                        origin=(0.35, 0, 0),
                    ),
                    fl.Slice(
                        name="slice_y",
                        normal=(0, 1, 0),
                        origin=(0, 0, 0),
                    ),
                ],
                output_fields=["T", "Mach"],
            ),
        ],
    )

case = project.run_case(params=params, name="CHT case from Python")

case.wait()

results = case.results

surface_heat_transfer = results.surface_heat_transfer.as_dataframe()
print(surface_heat_transfer)

cases = [case]

exclude = ["fluid/farfield", "solid/interface_fluid", "solid/adiabatic"]

front_camera_slice = FrontCamera(dimension=1, dimension_dir="width")
side_camera_slice = LeftCamera(pan_target=(0.35, 0, 0), dimension=2, dimension_dir="width")
front_right_top_camera = Camera(
    position=(-1, -1, 1), look_at=(0.35, 0, 0), dimension=1, dimension_dir="width"
)

x_slice_screenshot = Chart3D(
    section_title="Slice temperature at x=0.35",
    items_in_row=2,
    force_new_page=True,
    show="slices",
    include=["slice_x"],
    field="T",
    limits=(285 * u.K, 395 * u.K),
    camera=front_camera_slice,
    fig_name="slice_x",
)

y_slice_screenshot = Chart3D(
    section_title="Slice temperature at y=0",
    items_in_row=2,
    force_new_page=True,
    show="slices",
    include=["slice_y"],
    field="T",
    limits=(285 * u.K, 395 * u.K),
    camera=side_camera_slice,
    fig_name="slice_y",
)

surface_screenshot = Chart3D(
    section_title="Surface temperature",
    items_in_row=2,
    force_new_page=True,
    show="boundaries",
    field="T",
    limits=(285 * u.K, 395 * u.K),
    exclude=exclude,
    camera=front_right_top_camera,
)

report = ReportTemplate(
    title="CHT results screenshots",
    items=[Summary(), Inputs(), x_slice_screenshot, y_slice_screenshot, surface_screenshot],
    settings=Settings(dpi=150),
)

report = report.create_in_cloud(
    "CHT, dpi=default",
    cases,
    solver_version=__solver_version__,
)

report.wait()
report.download("report.pdf")
