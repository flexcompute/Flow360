import flow360 as fl
from flow360 import log, u
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
from flow360.user_config import UserConfig

log.set_logging_level("DEBUG")
UserConfig.set_profile("auto_test_1")


fl.Env.preprod.active()

case1 = fl.Case("case-ae75de95-bc8d-4f12-8607-6fec7763d36a")
case2 = fl.Case("case-713d66b6-4fc5-49ed-a5ea-2850d0d8d2bb")
case3 = fl.Case("case-1c8f54e9-c3cb-415f-bd58-54a37b4baaca")


cases = [case1, case2, case3]
freestream_surfaces = ["24", "25"]
slip_wall_surfaces = ["27", "28", "29", "30", "58"]
exclude = ["26", "33", "57", "59"]
size = "225M"

# # dev:
# fl.Env.dev.active()
# case1 = fl.Case("case-1e3f910e-337b-4e69-a313-0a42cefef7dc")  # // 0 Cpt, wall shear stress,
# case2 = fl.Case("case-f71193a2-0ce1-40b5-a087-c456fcf0bb21")  # // 2
# case3 = fl.Case("case-4dc6f67a-1bce-4152-b523-5822e09ce122")  # // 4


# cases = [case1]
# freestream_surfaces = ["blk-1/WT_side1", "blk-1/WT_side2", "blk-1/WT_inlet", "blk-1/WT_outlet"]
# slip_wall_surfaces = ["blk-1/WT_ceiling", "blk-1/WT_ground_front", "blk-1/WT_ground"]
# exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]
# size = "5.7M"


exclude += freestream_surfaces + slip_wall_surfaces

SOLVER_VERSION = "reportPipeline-24.10.13"


top_camera = Camera(
    position=(0, 0, 1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 1, 0),
    dimension=5,
    dimension_dir="width",
)
top_camera_slice = Camera(
    position=(0, 0, 1),
    look_at=(0, 0, 0),
    pan_target=(2.5, 0, 0),
    up=(0, 1, 0),
    dimension=8,
    dimension_dir="width",
)
side_camera = Camera(
    position=(0, -1, 0),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 0, 1),
    dimension=5,
    dimension_dir="width",
)
side_camera_slice = Camera(
    position=(0, -1, 0),
    look_at=(0, 0, 0),
    pan_target=(2.5, 0, 1.5),
    up=(0, 0, 1),
    dimension=8,
    dimension_dir="width",
)
back_camera = Camera(position=(1, 0, 0), up=(0, 0, 1), dimension=2.5, dimension_dir="width")
front_camera = Camera(position=(-1, 0, 0), up=(0, 0, 1), dimension=2.5, dimension_dir="width")
bottom_camera = Camera(
    position=(0, 0, -1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, -1, 0),
    dimension=5,
    dimension_dir="width",
)
front_left_bottom_camera = Camera(
    position=(-1, -1, -1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 0, 1),
    dimension=5,
    dimension_dir="width",
)
rear_right_bottom_camera = Camera(
    position=(1, 1, -1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 0, 1),
    dimension=6,
    dimension_dir="width",
)
front_left_top_camera = Camera(
    position=(-1, -1, 1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 0, 1),
    dimension=6,
    dimension_dir="width",
)
rear_left_top_camera = Camera(
    position=(1, -1, 1),
    look_at=(0, 0, 0),
    pan_target=(1.5, 0, 0),
    up=(0, 0, 1),
    dimension=6,
    dimension_dir="width",
)

cameras_geo = [
    top_camera,
    side_camera,
    back_camera,
    bottom_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]


limits_cp = [(-1, 1), (-1, 1), (-1, 1), (-0.3, 0), (-0.3, 0), (-1, 1), (-1, 1), (-1, 1)]
cameras_cp = [
    front_camera,
    front_left_top_camera,
    side_camera,
    rear_left_top_camera,
    back_camera,
    bottom_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]


avg = Average(fraction=0.1)
CD = DataItem(data="surface_forces/totalCD", exclude=exclude, title="CD", operations=avg)

CL = DataItem(data="surface_forces/totalCL", exclude=exclude, title="CL", operations=avg)

CDA = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CD*area",
    variables=[Variable(name="area", data="params.reference_geometry.area")],
    operations=[Expression(expr="totalCD * area"), avg],
)

CLf = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CLf",
    operations=[Expression(expr="1/2*totalCL + totalCMy"), avg],
)

CLr = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CLr",
    operations=[Expression(expr="1/2*totalCL - totalCMy"), avg],
)

CFy = DataItem(data="surface_forces/totalCFy", exclude=exclude, title="CS", operations=avg)

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
        exclude=exclude,
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
        exclude=exclude,
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
        exclude=exclude,
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
        field="velocity",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
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
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=side_camera_slice,
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
        field="velocity",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
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
        exclude=exclude,
        limits=(0, 5),
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
        exclude=exclude,
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
        exclude=exclude,
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
        exclude=exclude,
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
    f"{size}-{len(cases)}cases-slices-using-groups-Cpt, Cpx, wallShear, dpi=default",
    cases,
    solver_version=SOLVER_VERSION,
)

report.wait()
report.download("report.pdf")
