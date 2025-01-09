import flow360 as fl
from flow360 import log
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    Camera,
    Chart2D,
    Chart3D,
    Inputs,
    Summary,
    Table,
)
from flow360.plugins.report.utils import Average, DataItem, Delta, Expression
from flow360.user_config import UserConfig

log.set_logging_level("DEBUG")
UserConfig.set_profile("auto_test_1")


fl.Env.preprod.active()


case1 = fl.Case("case-bbf9a4dc-f5f7-42ee-bfe8-8905d9e45386")  # DrivAer
case2 = fl.Case("case-739c1c4d-aeb1-4c6e-af0c-b3ffb66c7a63")  # DrivAer beta=2
case3 = fl.Case("case-84582ff7-e421-4b6d-bb02-9af4f897764c")  # DrivAer beta=4

SOLVER_VERSION = "reportPipeline-24.10.4"


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
    pan_target=(1.5, 0, 0),
    up=(0, 1, 0),
    dimension=10,
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
    pan_target=(1.5, 0, 1.5),
    up=(0, 0, 1),
    dimension=10,
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
    dimension=5,
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
front_side_bottom_camera = Camera(
    position=(-1, -1, -1),
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

exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]


avg = Average(fraction=0.1)
CD = DataItem(data="surface_forces/totalCD", exclude=exclude, title="CD", operations=avg)

CL = DataItem(data="surface_forces/totalCL", exclude=exclude, title="CL", operations=avg)


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

statistical_data = Table(
    data=[
        "params/reference_geometry/area",
        CD,
        Delta(data=CD),
        CL,
        CLf,
        CLr,
        CFy,
        "volume_mesh/stats/n_nodes",
        "params/time_stepping/max_steps",
    ],
    section_title="Statistical data",
)


report = ReportTemplate(
    title="Aerodynamic analysis of DrivAer",
    items=[
        Summary(),
        Inputs(),
        statistical_data,
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
        ),
        Chart2D(
            x="nonlinear_residuals/pseudo_step",
            y="nonlinear_residuals/1_momx",
            section_title=None,
            fig_name="residuals",
        ),
        *[
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
        ],
        Chart3D(
            section_title="Slice velocity",
            items_in_row=2,
            force_new_page=True,
            show="slices",
            include=["y-slice through moment center"],
            field="velocity",
            limits=(0, 0.18),
            camera=side_camera_slice,
            fig_name="slice_y",
        ),
        Chart3D(
            section_title="Slice velocity",
            items_in_row=2,
            force_new_page=True,
            show="slices",
            include=["z-slice through moment center"],
            field="velocity",
            limits=(0, 0.18),
            camera=top_camera_slice,
            fig_name="slice_z",
        ),
        *[
            Chart3D(
                section_title="y+",
                items_in_row=2,
                show="boundaries",
                field="yPlus",
                exclude=exclude,
                limits=(0, 100),
                camera=camera,
                fig_name=f"yplus_{i}",
                caption=f"limits={(0, 100)}",
            )
            for i, camera in enumerate([top_camera, bottom_camera])
        ],
        *[
            Chart3D(
                section_title="Cp",
                items_in_row=2,
                show="boundaries",
                field="Cp",
                exclude=exclude,
                limits=limits,
                camera=camera,
                fig_name=f"cp_{i}",
                caption=f"limits={limits}",
            )
            for i, (limits, camera) in enumerate(zip(limits_cp, cameras_cp))
        ],
        Chart3D(
            section_title="Q-criterion",
            items_in_row=2,
            force_new_page=True,
            show="qcriterion",
            exclude=exclude,
            field="Mach",
            limits=(0, 0.18),
            fig_name="qcriterion",
        ),
    ],
)

report = report.create_in_cloud(
    "running_report_from_python", [case1, case2, case3], solver_version=SOLVER_VERSION
)

report.wait()
report.download("report.pdf")
