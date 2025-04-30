"""This script is used by sweep_launch_template.py to create a report."""

from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    BottomCamera,
    Camera,
    Chart2D,
    Chart3D,
    FrontCamera,
    FrontLeftTopCamera,
    Inputs,
    LeftCamera,
    NonlinearResiduals,
    PatternCaption,
    RearCamera,
    RearRightBottomCamera,
    Settings,
    SubsetLimit,
    Summary,
    Table,
    TopCamera,
)
from flow360.plugins.report.utils import Average, DataItem
from flow360.version import __solver_version__


def generate_report(
    cases,
    params,
    include_geometry: bool = False,
    include_general_tables: bool = False,
    include_residuals: bool = False,
    include_cfl: bool = False,
    include_forces_moments_table: bool = False,
    include_forces_moments_charts: bool = False,
    include_forces_moments_alpha_charts: bool = False,
    include_forces_moments_beta_charts: bool = False,
    include_cf_vec: bool = False,
    include_cp: bool = False,
    include_yplus: bool = False,
    include_qcriterion: bool = False,
):
    items = []

    freestream_surfaces = ["fluid/farfield"]

    exclude = freestream_surfaces

    top_camera = TopCamera(pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="height")
    bottom_camera = BottomCamera(pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="height")
    front_camera = FrontCamera(pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="width")
    rear_camera = RearCamera(pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="width")
    left_camera = LeftCamera(pan_target=(3.5, 0, -0.5), dimension=10, dimension_dir="width")
    right_camera = Camera(
        pan_target=(3.5, 0, -0.5),
        position=(0.0, -1.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        dimension=10,
        dimension_dir="width",
    )
    front_left_top_camera = FrontLeftTopCamera(
        pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="width"
    )
    rear_right_bottom_camera = RearRightBottomCamera(
        pan_target=(3.5, 0, -0.5), dimension=15, dimension_dir="width"
    )

    if params.time_stepping.type_name == "Unsteady":
        step_type = "physical_step"
    else:
        step_type = "pseudo_step"

    geo_cameras = [
        top_camera,
        bottom_camera,
        front_camera,
        rear_camera,
        left_camera,
        right_camera,
        front_left_top_camera,
        rear_right_bottom_camera,
    ]

    geo_camera_names = [
        "top_camera",
        "bottom_camera",
        "front_camera",
        "rear_camera",
        "left_camera",
        "right_camera",
        "front_left_top_camera",
        "rear_right_bottom_camera",
    ]

    avg = Average(fraction=0.1)

    force_list = [
        "CD",
        "CL",
        "CFx",
        "CFy",
        "CFz",
        "CMx",
        "CMy",
        "CMz",
    ]

    for model in params.models:
        if model.type == "Fluid":
            turbulence_solver = model.turbulence_model_solver.type_name

    cfl_list = ["0_NavierStokes_cfl", f"1_{turbulence_solver}_cfl"]

    CD = DataItem(data="surface_forces/totalCD", exclude=exclude, title="CD", operations=avg)
    CL = DataItem(data="surface_forces/totalCL", exclude=exclude, title="CL", operations=avg)
    CFX = DataItem(data="surface_forces/totalCFx", exclude=exclude, title="CFx", operations=avg)
    CFY = DataItem(data="surface_forces/totalCFy", exclude=exclude, title="CFy", operations=avg)
    CFZ = DataItem(data="surface_forces/totalCFz", exclude=exclude, title="CFz", operations=avg)
    CMX = DataItem(data="surface_forces/totalCMx", exclude=exclude, title="CMx", operations=avg)
    CMY = DataItem(data="surface_forces/totalCMy", exclude=exclude, title="CMy", operations=avg)
    CMZ = DataItem(data="surface_forces/totalCMz", exclude=exclude, title="CMz", operations=avg)

    table_data = [
        CD,
        CL,
        CFX,
        CFY,
        CFZ,
        CMX,
        CMY,
        CMZ,
    ]

    if include_geometry:
        geometry_screenshots = [
            Chart3D(
                section_title="Geometry",
                items_in_row=2,
                force_new_page=True,
                show="boundaries",
                camera=front_left_top_camera,
                exclude=exclude,
                fig_name="Geometry_view",
            )
        ]
        items.extend(geometry_screenshots)

    if include_general_tables:
        items.append(Summary())
        items.append(Inputs())

    if include_forces_moments_table:
        table = Table(
            data=table_data,
            section_title="Quantities of interest",
        )
        items.append(table)

    if include_residuals:
        residual_charts = NonlinearResiduals(
            force_new_page=True,
            section_title="Nonlinear residuals",
            fig_name=f"nonlin-res_fig"
        )
        items.append(residual_charts)

    if include_cfl and params.time_stepping.CFL.type == "adaptive":
        cfl_charts = [
            Chart2D(
                x=f"cfl/{step_type}",
                y=f"cfl/{cfl}",
                force_new_page=True,
                section_title="CFL",
                fig_name=f"{cfl}_fig",
            )
            for cfl in cfl_list
        ]
        items.extend(cfl_charts)

    if include_forces_moments_charts:
        force_charts = [
            Chart2D(
                x=f"surface_forces/{step_type}",
                y=f"surface_forces/total{force}",
                force_new_page=True,
                section_title="Forces/Moments",
                fig_name=f"{force}_fig",
                exclude=exclude,
                ylim=SubsetLimit(subset=(0.5, 1), offset=0.25)
            )
            for force in force_list
        ]
        items.extend(force_charts)

    if include_forces_moments_alpha_charts:
        force_alpha_charts = [
            Chart2D(
                x=f"params/operating_condition/alpha",
                y=f"total_forces/averages/{force}",
                force_new_page=True,
                section_title="Averaged Forces/Moments against alpha",
                fig_name=f"{force}_alpha_fig"
            )
            for force in force_list
        ]
        items.extend(force_alpha_charts)

    if include_forces_moments_beta_charts:
        force_beta_charts = [
            Chart2D(
                x=f"params/operating_condition/beta",
                y=f"total_forces/averages/{force}",
                force_new_page=True,
                section_title="Averaged Forces/Moments against beta",
                fig_name=f"{force}_beta_fig"
            )
            for force in force_list
        ]
        items.extend(force_beta_charts)

    if include_yplus:
        y_plus_screenshots = [
            Chart3D(
                caption=PatternCaption(pattern=f"y+_{camera_name}_[case.name]"),
                show="boundaries",
                field="yPlus",
                exclude=exclude,
                limits=(0, 5),
                camera=camera,
                fig_name=f"yplus_{camera_name}_fig",
                fig_size=1,
            )
            for camera_name, camera in zip(geo_camera_names, geo_cameras)
        ]
        items.extend(y_plus_screenshots)

    if include_cp:
        cp_screenshots = [
            Chart3D(
                caption=PatternCaption(pattern=f"Cp_{camera_name}_[case.name]"),
                show="boundaries",
                field="Cp",
                exclude=exclude,
                limits=(-1, 1),
                camera=camera,
                fig_name=f"cp_{camera_name}_fig",
                fig_size=1,
            )
            for camera_name, camera in zip(geo_camera_names, geo_cameras)
        ]
        items.extend(cp_screenshots)

    if include_cf_vec:
        cfvec_screenshots = [
            Chart3D(
                caption=PatternCaption(pattern=f"Cf_vec_{camera_name}_[case.name]"),
                show="boundaries",
                field="CfVec",
                mode="lic",
                exclude=exclude,
                limits=(0, 0.025),
                camera=camera,
                fig_name=f"cfvec_{camera_name}_fig",
                fig_size=1,
            )
            for camera_name, camera in zip(geo_camera_names, geo_cameras)
        ]
        items.extend(cfvec_screenshots)

    if include_qcriterion:
        qcriterion_screenshots = [
            Chart3D(
                caption=PatternCaption(pattern=f"Isosurface_q_criterion_{camera_name}_[case.name]"),
                show="isosurface",
                iso_field="qcriterion",
                exclude=exclude,
                limits=(0, 0.8),
                camera=camera,
                fig_name=f"qcriterion_{camera_name}_fig",
                fig_size=1,
            )
            for camera_name, camera in zip(geo_camera_names, geo_cameras)
        ]
        items.extend(qcriterion_screenshots)

    report = ReportTemplate(
        title="Sweep Template Report",
        items=items,
        settings=Settings(dpi=150),
    )

    report = report.create_in_cloud(
        "sweep-script-report",
        cases,
        solver_version=__solver_version__,
    )

    report.wait()
    report.download("report.pdf")
