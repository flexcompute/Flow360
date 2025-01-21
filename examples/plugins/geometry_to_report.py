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
from flow360.plugins.report.utils import Average, DataItem, Delta, Expression, Variable


project_id = None # if running for the first time, than replace it with project ID to avoid re-creation of projects
# project_id = "prj-b5a0ae52-14c7-4f0c-813b-542763f993a2"

if project_id is not None:
    project = fl.Project.from_cloud(project_id)
else:
    project = fl.Project.from_file(
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
                    name="Wall",
                ),
                fl.Freestream(
                    surfaces=[far_field_zone.farfield],
                    name="Freestream",
                ),
            ],
            user_defined_fields=[
                fl.UserDefinedField(
                    name="velocityMag",
                    expression="double velocity_vec[3];"
                    + "velocity_vec[0] = primitiveVars[1];"
                    + "velocity_vec[1] = primitiveVars[2];"
                    + "velocity_vec[2] = primitiveVars[3];"
                    + "velocityMag = magnitude(velocity_vec)",
                ),
                fl.UserDefinedField(
                    name="velocityVec",
                    expression="velocityVec[0] = primitiveVars[1];"
                    + "velocityVec[1] = primitiveVars[2];"
                    + "velocityVec[2] = primitiveVars[3];",
                ),
                fl.UserDefinedField(name="wallShearMag", expression="magnitude(wallShearStress);"),
                fl.UserDefinedField(
                    name="Cpx",
                    expression="double prel = primitiveVars[4] - pressureFreestream;"
                    + "double PressureForce_X = prel * nodeNormals[0]; "
                    + "Cpx = PressureForce_X / (0.5 * MachRef * MachRef) / magnitude(nodeNormals);",
                ),
            ],
            outputs=[
                fl.SurfaceOutput(
                    name="surface",
                    surfaces=geo["*"],
                    output_fields=[
                        "Cp",
                        "Cf",
                        "yPlus",
                        "CfVec",
                        "primitiveVars",
                        "wallShearMag",
                        "Cpx",
                    ],
                ),
                fl.SliceOutput(
                    name="slices",
                    entities=[
                        *[
                            fl.Slice(
                                name=f"slice_y_{name}",
                                normal=(0, 1, 0),
                                origin=(0, y, 0),
                            )
                            for name, y in zip(
                                ["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8]
                            )
                        ],
                        *[
                            fl.Slice(
                                name=f"slice_z_{name}",
                                normal=(0, 0, 1),
                                origin=(0, 0, z),
                            )
                            for name, z in zip(
                                ["neg0_2", "0", "0_2", "0_4", "0_6", "0_8"],
                                [-0.2, 0, 0.2, 0.4, 0.6, 0.8],
                            )
                        ],
                    ],
                    output_fields=["velocityMag", "velocityVec"],
                ),
                fl.IsosurfaceOutput(
                    name="cpt",
                    output_fields=["Cp", "Mach"],
                    isosurfaces=[
                        fl.Isosurface(
                            name="isosurface-cpt",
                            iso_value=-1,
                            field="Cpt",
                        ),
                    ],
                ),
            ],
        )
    return params


cases: list[fl.Case] = []
for alpha in [0, 2, 4]:
    case = project.run_case(
        params=simulation_params(alpha),
        name=f"Case for report, alpha={alpha}"
    )
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
top_camera_slice = Camera(
    position=(0, 0, 1),
    look_at=(0, 0, 0),
    pan_target=(4, 0, 0),
    up=(0, 1, 0),
    dimension=18,
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
side_camera_slice = Camera(
    position=(0, -1, 0),
    look_at=(0, 0, 0),
    pan_target=(4, 0, 0),
    up=(0, 0, 1),
    dimension=18,
    dimension_dir="width",
)
side_camera_slice_lic = side_camera_slice

back_camera = Camera(position=(1, 0, 0), up=(0, 0, 1), dimension=12, dimension_dir="width")
front_camera = Camera(position=(-1, 0, 0), up=(0, 0, 1), dimension=12, dimension_dir="width")
bottom_camera = Camera(
    position=(0, 0, -1),
    look_at=(0, 0, 0),
    pan_target=(5, 0, 0),
    up=(0, -1, 0),
    dimension=15,
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
front_left_top_camera = Camera(
    position=(-1, -1, 1),
    look_at=(0, 0, 0),
    pan_target=(4, 0, 0),
    up=(0, 0, 1),
    dimension=15,
    dimension_dir="width",
)
rear_left_top_camera = Camera(
    position=(1, -1, 1),
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

limits_cp = [(-1, 1), (-1, 1), (-1, 1), (-0.3, 0), (-0.3, 0), (-1, 1), (-1, 1), (-1, 1)]
cameras_cp = [
    front_camera,
    front_left_top_camera,
    side_camera,
    rear_left_top_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]


avg = Average(fraction=0.1)
CD = DataItem(data="surface_forces/totalCD", title="CD", operations=avg)

CL = DataItem(data="surface_forces/totalCL", title="CL", operations=avg)

CDA = DataItem(
    data="surface_forces",
    title="CD*area",
    variables=[Variable(name="area", data="params.reference_geometry.area")],
    operations=[Expression(expr="totalCD * area"), avg],
)

CLf = DataItem(
    data="surface_forces",
    title="CLf",
    operations=[Expression(expr="1/2*totalCL + totalCMy"), avg],
)

CLr = DataItem(
    data="surface_forces",
    title="CLr",
    operations=[Expression(expr="1/2*totalCL - totalCMy"), avg],
)

CFy = DataItem(data="surface_forces/totalCFy", title="CS", operations=avg)

statistical_data = [
    "params/reference_geometry/area",
    CD,
    CDA,
    Delta(data=CD),
    CL,
    CLf,
    CLr,
    CFy,
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

cpt_screenshots = [
    Chart3D(
        section_title="Isosurface, Cpt=-1",
        items_in_row=2,
        force_new_page=True,
        show="isosurface",
        iso_field="Cpt",
        camera=camera,
    )
    for camera in cameras_cp
]

cfvec_screenshots = [
    Chart3D(
        section_title="CfVec",
        items_in_row=2,
        force_new_page=True,
        show="boundaries",
        field="CfVec",
        mode="lic",
        limits=(1e-4, 10),
        is_log_scale=True,
        camera=camera,
    )
    for camera in cameras_cp
]


y_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocityMag",
        limits=(0 * u.m / u.s, 100 * u.m / u.s),
        camera=side_camera_slice,
        fig_name=f"slice_y_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]


y_slices_lic_screenshots = [
    Chart3D(
        section_title=f"Slice velocity LIC y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocityVec",
        mode="lic",
        limits=(0 * u.m / u.s, 100 * u.m / u.s),
        camera=side_camera_slice_lic,
        fig_name=f"slice_y_vec_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]

z_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity z={z}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_z_{name}"],
        field="velocityMag",
        limits=(0 * u.m / u.s, 100 * u.m / u.s),
        camera=top_camera_slice,
        fig_name=f"slice_z_{name}",
    )
    for name, z in zip(["neg0_2", "0", "0_2", "0_4", "0_6", "0_8"], [-0.2, 0, 0.2, 0.4, 0.6, 0.8])
]

y_plus_screenshots = [
    Chart3D(
        section_title="y+",
        items_in_row=2,
        show="boundaries",
        field="yPlus",
        limits=(0, 100),
        camera=camera,
        fig_name=f"yplus_{i}",
    )
    for i, camera in enumerate([top_camera, bottom_camera])
]
cp_screenshots = [
    Chart3D(
        section_title="Cp",
        items_in_row=2,
        show="boundaries",
        field="Cp",
        limits=limits,
        camera=camera,
        fig_name=f"cp_{i}",
    )
    for i, (limits, camera) in enumerate(zip(limits_cp, cameras_cp))
]
cpx_screenshots = [
    Chart3D(
        section_title="Cpx",
        items_in_row=2,
        show="boundaries",
        field="Cpx",
        limits=(-0.3, 0.3),
        camera=camera,
        fig_name=f"cpx_{i}",
    )
    for i, camera in enumerate(cameras_cp)
]

wall_shear_screenshots = [
    Chart3D(
        section_title="Wall shear stress magnitude",
        items_in_row=2,
        show="boundaries",
        field="wallShearMag",
        limits=(0 * u.Pa, 5 * u.Pa),
        camera=camera,
        fig_name=f"wallShearMag_{i}",
    )
    for i, camera in enumerate(cameras_cp)
]

report = ReportTemplate(
    title="Aerodynamic analysis of DrivAer",
    items=[
        Summary(),
        Inputs(),
        statistical_table,
        Chart2D(
            x="x_slicing_force_distribution/X",
            y="x_slicing_force_distribution/totalCumulative_CD_Curve",
            fig_name="totalCumulative_CD_Curve",
            background="geometry",
        ),
        Chart2D(
            x="surface_forces/pseudo_step",
            y="surface_forces/totalCD",
            section_title="Drag Coefficient",
            fig_name="cd_fig",
            focus_x=(1 / 3, 1),
        ),
        *geometry_screenshots,
        *cpt_screenshots,
        *y_slices_screenshots,
        *y_slices_lic_screenshots,
        *z_slices_screenshots,
        *y_plus_screenshots,
        *cp_screenshots,
        *cpx_screenshots,
        *wall_shear_screenshots,
        *cfvec_screenshots,
    ],
    settings=Settings(dpi=150),
)


report = report.create_in_cloud(
    f"Geometry to report - Report, dpi=150",
    cases,
    solver_version="reportPipeline-24.10.14",
)

report.wait()
report.download("report.pdf")
