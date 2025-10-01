import flow360 as fl
from flow360 import u
from flow360.examples import Airplane
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    Camera,
    Chart2D,
    Chart3D,
    Inputs,
    Settings,
    Summary,
    Table,
)
from flow360.plugins.report.utils import Average, DataItem, Delta

project_id = None  # if running for the first time

# then replace it with your project ID to avoid re-creation of projects. You can find project ID on web GUI:
# project_id = "prj-...."

if project_id is not None:
    project = fl.Project.from_cloud(project_id)
else:
    project = fl.Project.from_geometry(
        Airplane.geometry, name="Python Project (Geometry, from file) - for Report"
    )

geo = project.geometry
geo.group_faces_by_tag("groupName")


def simulation_params(angle_of_attack):
    with fl.SI_unit_system:
        far_field_zone = fl.AutomatedFarfield()
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    surface_max_edge_length=1,
                ),
                volume_zones=[far_field_zone],
            ),
            reference_geometry=fl.ReferenceGeometry(area=1, moment_length=1),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100,
                alpha=angle_of_attack * fl.u.deg,
            ),
            time_stepping=fl.Steady(max_steps=1000),
            models=[
                fl.Wall(
                    surfaces=[geo["*"]],
                ),
                fl.Freestream(
                    surfaces=[far_field_zone.farfield],
                ),
            ],
            outputs=[
                fl.SurfaceOutput(
                    surfaces=geo["*"],
                    output_fields=[
                        "Cp",
                        "Cf",
                        "yPlus",
                        "CfVec",
                        "primitiveVars",
                    ],
                ),
            ],
        )
    return params


cases: list[fl.Case] = []
for alpha in [0, 2, 4]:
    case = project.run_case(params=simulation_params(alpha), name=f"Case for report, alpha={alpha}")
    cases.append(case)

[print(case.short_description()) for case in cases]

# waiting explicitly for all the cases to finish (report pipeline will not wait for cases)
[case.wait() for case in cases]


top_camera = Camera(
    position=(0, 0, 1),
    look_at=(0, 0, 0),
    pan_target=(5, 0, 0),
    up=(0, 1, 0),
    dimension=15,
    dimension_dir="width",
)
side_camera = Camera(
    position=(0, -1, 0),
    look_at=(0, 0, 0),
    pan_target=(5, 0, 0),
    up=(0, 0, 1),
    dimension=12,
    dimension_dir="width",
)
front_left_bottom_camera = Camera(
    position=(-1, -1, -1),
    look_at=(0, 0, 0),
    pan_target=(4, 0, 0),
    up=(0, 0, 1),
    dimension=15,
    dimension_dir="width",
)
rear_right_bottom_camera = Camera(
    position=(1, 1, -1),
    look_at=(0, 0, 0),
    pan_target=(4, 0, 0),
    up=(0, 0, 1),
    dimension=15,
    dimension_dir="width",
)

cameras_geo = [
    top_camera,
    side_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]

avg = Average(fraction=0.1)

CL = DataItem(data="surface_forces/totalCL", title="CL", operations=avg)

CD = DataItem(data="surface_forces/totalCD", title="CD", operations=avg)

statistical_data = [
    "params/operating_condition/alpha",
    "params/reference_geometry/area",
    CL,
    Delta(data=CL),
    CD,
    "volume_mesh/stats/n_nodes",
    "params/time_stepping/max_steps",
]
statistical_table = Table(
    data=statistical_data,
    section_title="Statistical data",
    formatter=[
        (
            None
            if d
            in [
                "params/reference_geometry/area",
                "volume_mesh/stats/n_nodes",
                "params/time_stepping/max_steps",
            ]
            else ".4f"
        )
        for d in statistical_data
    ],
)

geometry_screenshots = [
    Chart3D(
        section_title="Geometry",
        items_in_row=2,
        force_new_page=True,
        show="boundaries",
        camera=camera,
        fig_name=f"geo_{i}",
    )
    for i, camera in enumerate(cameras_geo)
]

report = ReportTemplate(
    title="Geometry to report",
    items=[
        Summary(),
        Inputs(),
        statistical_table,
        Chart2D(
            x="total_forces/pseudo_step",
            y="total_forces/CL",
            section_title="Lift Coefficient",
            fig_name="cl_fig",
            focus_x=(1 / 3, 1),
        ),
        *geometry_screenshots,
    ],
    settings=Settings(dpi=150),
)


report = report.create_in_cloud(
    f"Geometry to report - Report, dpi=150",
    cases,
)

report.wait()
report.download("report.pdf")
