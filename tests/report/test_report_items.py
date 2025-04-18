import os

import numpy as np
import pandas as pd
import pytest
from pylatex import Document
from pylatex.utils import bold, escape_latex

from flow360 import Case, u
from flow360.component.case import CaseMeta
from flow360.component.resource_base import local_metadata_builder
from flow360.component.utils import LocalResourceCache
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.report_doc import ReportDoc
from flow360.plugins.report.report_items import (
    Camera,
    Chart2D,
    Chart3D,
    FixedRangeLimit,
    ManualLimit,
    NonlinearResiduals,
    PatternCaption,
    PlotModel,
    SubsetLimit,
    Table,
    human_readable_formatter,
)
from flow360.plugins.report.utils import (
    Average,
    DataItem,
    Delta,
    Expression,
    GetAttribute,
)


@pytest.fixture
def here():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def cases(here):

    case_ids = [
        "case-11111111-1111-1111-1111-111111111111",
        "case-2222222222-2222-2222-2222-2222222222",
    ]

    cache = LocalResourceCache()

    cases = []
    for cid in case_ids:
        case_meta = CaseMeta(
            caseId=cid,
            name=f"{cid}-name",
            status="completed",
            userId="user-id",
            caseMeshId="vm-11111111-1111-1111-1111-111111111111",
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        )
        case = Case.from_local_storage(os.path.join(here, "..", "data", cid), case_meta)
        cases.append(case)

    vm_id = "vm-11111111-1111-1111-1111-111111111111"
    vm = VolumeMeshV2.from_local_storage(
        mesh_id=vm_id,
        local_storage_path=os.path.join(here, "..", "data", vm_id),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id=vm_id,
                name="DrivAer mesh",
                cloud_path_prefix="s3://flow360meshes-v1/users/user-id",
            )
        ),
    )
    cache.add(vm)

    return cases


@pytest.fixture
def residual_plot_model(here):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    residual_data = pd.read_csv(
        os.path.join(
            here,
            "..",
            "data",
            "case-11111111-1111-1111-1111-111111111111",
            "results",
            "nonlinear_residual_v2.csv",
        ),
        skipinitialspace=True,
    )

    x_data = [list(residual_data["pseudo_step"]) for _ in residuals_sa]
    y_data = [list(residual_data[res]) for res in residuals_sa]

    x_label = "pseudo_step"

    return PlotModel(x_data=x_data, y_data=y_data, x_label=x_label, y_label="none")


@pytest.fixture
def two_var_two_cases_plot_model(here, cases):
    loads = ["CL", "CD"]

    x_data = []
    y_data = []
    legend = []
    for case in cases:
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.info.id, "results", "total_forces_v2.csv"),
            skipinitialspace=True,
        )

        for load in loads:
            x_data.append(list(load_data["pseudo_step"]))
            y_data.append(list(load_data[load]))

    y_label = "value"
    x_label = "pseudo_step"

    return PlotModel(x_data=x_data, y_data=y_data, x_label=x_label, y_label=y_label)


@pytest.mark.parametrize(
    "value,expected",
    [
        # Large values (millions)
        (225422268, "225M"),  # Large number well into millions
        (1000000, "1M"),  # Exactly 1 million
        (9999999, "10M"),  # Just under 10 million, rounds to 10M
        (25400000, "25M"),  # Between 10 and 100 million
        # Thousands
        (22542, "23k"),  # Between 10k and 100k => one decimal
        (225422, "225k"),  # Over 100k => no decimals
        (2254, "2.3k"),  # Under 10k => one decimal
        (1000, "1k"),  # Exactly 1k
        # Less than 1000
        (225.4, "225.4"),  # No suffix, up to two decimals
        (2.345, "2.345"),  # No change
        (2, "2"),  # Whole number <1000
        (0.5, "0.5"),  # Decimal less than 1
        (0.123456, "0.123456"),  # no change
        # Negative values
        (-225422268, "-225M"),  # Negative large number
        (-22542, "-23k"),
        (-2254, "-2.3k"),
        (-225.4, "-225.4"),
        (-2.345, "-2.345"),
        # Non-numeric
        ("abc", "abc"),
        (None, "None"),
    ],
)
def test_human_readable_formatter(value, expected):
    assert human_readable_formatter(value) == expected


def test_cameras():
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
        for name, z in zip(
            ["neg0_2", "0", "0_2", "0_4", "0_6", "0_8"], [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
        )
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
            *geometry_screenshots,
            *cpt_screenshots,
            *cfvec_screenshots,
            *y_slices_screenshots,
            *y_slices_lic_screenshots,
            *z_slices_screenshots,
            *y_plus_screenshots,
            *cp_screenshots,
            *cpx_screenshots,
            *wall_shear_screenshots,
        ],
    )
    assert report


def test_operation():
    with pytest.raises(ValueError, match="Value error, One of"):
        Average()
    with pytest.raises(
        ValueError, match="start_step and start_time cannot be specified at the same time"
    ):
        Average(start_step=1, start_time=2)

    Average(start_step=1)
    Average(start_time=2)
    Average(fraction=0.5)


def test_tables(cases):
    context = ReportContext(
        cases=cases,
        doc=Document(),
        data_storage=".",
    )

    freestream_surfaces = ["blk-1/WT_side1", "blk-1/WT_side2", "blk-1/WT_inlet", "blk-1/WT_outlet"]
    slip_wall_surfaces = ["blk-1/WT_ceiling", "blk-1/WT_ground_front", "blk-1/WT_ground"]
    exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]
    exclude += freestream_surfaces + slip_wall_surfaces

    include = ["blk-1/wheel_rim", "blk-1/BODY", "blk-1/wheel_tire"]

    filtering = [dict(include=include), dict(exclude=exclude)]

    for filter in filtering:
        print(f"testing: {filter=}")

        avg = Average(fraction=0.1)
        CD = DataItem(data="surface_forces/totalCD", title="CD", operations=avg, **filter)

        CL = DataItem(data="surface_forces/totalCL", title="CL", operations=avg, **filter)

        CLCompare = DataItem(
            data="surface_forces",
            title="CL_compare",
            operations=[Expression(expr="totalCL - 1.5467"), avg],
            **filter,
        )

        CDCompare = DataItem(
            data="surface_forces",
            title="CD_compare",
            operations=[Expression(expr="totalCD - 0.02100"), avg],
            **filter,
        )

        CLf = DataItem(
            data="surface_forces",
            title="CLf",
            operations=[Expression(expr="1/2*totalCL + totalCMy"), avg],
            **filter,
        )

        CLr = DataItem(
            data="surface_forces",
            title="CLr",
            operations=[Expression(expr="1/2*totalCL - totalCMy"), avg],
            **filter,
        )

        OWL = DataItem(
            data="volume_mesh/bounding_box",
            title="OWL",
            operations=[GetAttribute(attr_name="length")],
            **filter,
        )

        OWW = DataItem(
            data="volume_mesh/bounding_box",
            title="OWW",
            operations=[GetAttribute(attr_name="width")],
            **filter,
        )

        OWH = DataItem(
            data="volume_mesh/bounding_box",
            title="OWH",
            operations=[GetAttribute(attr_name="height")],
            **filter,
        )
        CFy = DataItem(data="surface_forces/totalCFy", title="CS", operations=avg, **filter)

        statistical_data = Table(
            data=[
                "params/reference_geometry/area",
                CD,
                Delta(data=CD),
                CL,
                CLCompare,
                CDCompare,
                CLf,
                CLr,
                CFy,
                "volume_mesh/stats/n_nodes",
                "params/time_stepping/max_steps",
                OWL,
                OWW,
                OWH,
            ],
            section_title="Statistical data",
        )

        table_df = statistical_data.to_dataframe(context)
        table_df["Case No."] = table_df["Case No."].astype("Int64")
        table_df["area"] = table_df["area"].astype(str)

        print(table_df)

        expected_data = {
            "Case No.": [1, 2],
            "area": ["2.17 m**2", "2.17 m**2"],
            "CD": [0.279249, 0.288997],
            "Delta CD": [0.000000, 0.009748],
            "CL": [0.145825, 0.169557],
            "CL_compare": [-1.400875, -1.377143],
            "CD_compare": [0.258249, 0.267997],
            "CLf": [-0.050186, -0.157447],
            "CLr": [0.196011, 0.327003],
            "CS": [-0.002243102563079525, -0.0763879853938102],
            "n_nodes": [5712930, 5712930],
            "max_steps": [2000, 2000],
            "OWL": [4.612806, 4.612806],
            "OWW": [2.029983, 2.029983],
            "OWH": [1.405979, 1.405979],
        }
        df_expected = pd.DataFrame(expected_data)
        df_expected["Case No."] = df_expected["Case No."].astype("Int64")
        print(df_expected)

        pd.testing.assert_frame_equal(table_df, df_expected)


def test_calculate_y_lim(cases, here):
    chart = Chart2D(
        x="total_forces/pseudo_step",
        y="total_forces/CD",
    )
    case = cases[0]
    case_data = pd.read_csv(
        os.path.join(here, "..", "data", case.id, "results", "total_forces_v2.csv")
    )
    x_series_list = [case_data[" pseudo_step"].to_list()]
    y_series_list = [case_data[" CD"].to_list()]

    chart.ylim = (0.3, 0.4)
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.3
    assert ymax == 0.4

    chart.ylim = ManualLimit(lower=0.3, upper=0.4)
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.3
    assert ymax == 0.4

    chart.ylim = SubsetLimit(subset=(0.5, 0.9), offset=0.25)
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.34713380202529676
    assert ymax == 0.3558530166937262

    chart.ylim = None
    chart.focus_x = (0.5, 0.9)
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.34713380202529676
    assert ymax == 0.3558530166937262

    chart.focus_x = None
    chart.ylim = FixedRangeLimit(fixed_range=0.1)
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.306161580155019
    assert ymax == 0.40616158015501896

    chart.ylim = FixedRangeLimit(
        fixed_range=0.1, center_strategy="last_percent", center_fraction=0.7
    )
    ymin, ymax = chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list)
    assert ymin == 0.3528853000874695
    assert ymax == 0.45288530008746947

    chart.ylim = None
    assert (
        chart._calculate_ylimits(x_series_list=x_series_list, y_series_list=y_series_list) == None
    )

    with pytest.raises(ValueError, match="Fields ylim and focus_x cannot be used together."):
        chart.ylim = (0.5, 0.9)
        chart.focus_x = (0.5, 0.9)
        chart._calculate_ylimits(
            x_series_list=x_series_list,
            y_series_list=y_series_list,
        )


def test_dimensioned_limits(cases):

    case = cases[0]

    chart = Chart3D(
        field="velocity",
        show="boundaries",
        limits=(0, 0.3),
    )
    assert chart.limits == (0, 0.3)

    converted_limits = chart._get_limits(case)
    assert converted_limits == (0, 0.3)

    chart = Chart3D(
        field="velocity",
        show="boundaries",
        limits=(0 * u.m / u.s, 100 * u.m / u.s),
    )
    assert chart.limits == (0 * u.m / u.s, 100 * u.m / u.s)

    converted_limits = chart._get_limits(case)
    assert converted_limits == (0, 0.2938635365101296)

    chart = Chart3D(
        field="velocity_m_per_s",
        show="boundaries",
        limits=(0 * u.m / u.s, 100 * u.m / u.s),
    )
    assert chart.limits == (0 * u.m / u.s, 100 * u.m / u.s)

    converted_limits = chart._get_limits(case)
    assert converted_limits == (0, 100)

    chart = Chart3D(
        field="velocity_m_per_s",
        show="boundaries",
        limits=(0 * u.km / u.hr, 72 * u.km / u.hr),
    )
    assert chart.limits == (0 * u.km / u.hr, 72 * u.km / u.hr)

    converted_limits = chart._get_limits(case)
    assert converted_limits == (0, 20)

    chart = Chart3D(
        field="velocity_m_per_s",
        show="boundaries",
        limits=(0, 10),
    )
    assert chart.limits == (0, 10)

    converted_limits = chart._get_limits(case)
    assert converted_limits == (0, 10)


def test_2d_caption_validity(cases):
    chart = Chart2D(
        x="total_forces/pseudo_step",
        y="total_forces/CD",
    )

    with pytest.raises(
        ValueError, match="List of captions and items_in_row cannot be used together."
    ):
        Chart2D(
            x="total_forces/pseudo_step",
            y="total_forces/CD",
            items_in_row=2,
            caption=["Caption 1", "Caption 2"],
        )

    with pytest.raises(
        ValueError, match="PatternCaption and items_in_row cannot be used together."
    ):
        Chart2D(
            x="total_forces/pseudo_step",
            y="total_forces/CD",
            items_in_row=2,
            caption=PatternCaption(pattern="This is case: [case.name] with ID: [case.id]"),
        )

    with pytest.raises(
        ValueError,
        match="List of captions is only supported for Chart2D when separate_plots is True.",
    ):
        Chart2D(
            x="total_forces/pseudo_step", y="total_forces/CD", caption=["Caption 1", "Caption 2"]
        )

    with pytest.raises(
        ValueError,
        match="PatternCaption is only supported for Chart2D when separate_plots is True.",
    ):
        Chart2D(
            x="total_forces/pseudo_step",
            y="total_forces/CD",
            caption=PatternCaption(pattern="This is case: [case.name] with ID: [case.id]"),
        )

    with pytest.raises(
        ValueError, match="Caption list is not the same length as the list of cases."
    ):
        chart.separate_plots = True
        chart.caption = ["Caption 1", "Caption 2", "Caption 3"]
        chart._check_caption_validity(cases)


def test_2d_caption(cases):
    chart = Chart2D(
        x="total_forces/pseudo_step",
        y="total_forces/CD",
    )

    chart.caption = "This is a caption."
    assert chart._handle_2d_caption() == "This is a caption."

    chart.separate_plots = True
    chart.caption = ["Caption 1", "Caption 2"]
    assert chart._handle_2d_caption(case_number=0) == "Caption 1"
    assert chart._handle_2d_caption(case_number=1) == "Caption 2"

    chart.caption = PatternCaption(pattern="This is case: [case.name] with ID: [case.id]")
    assert chart._handle_2d_caption(case=cases[0]) == escape_latex(
        "This is case: case-11111111-1111-1111-1111-111111111111-name with ID: case-11111111-1111-1111-1111-111111111111"
    )
    assert chart._handle_2d_caption(case=cases[1]) == escape_latex(
        "This is case: case-2222222222-2222-2222-2222-2222222222-name with ID: case-2222222222-2222-2222-2222-2222222222"
    )

    chart_selected_cases = Chart2D(
        x="total_forces/pseudo_step", y="total_forces/CD", select_indices=[1]
    )

    assert (
        chart_selected_cases._handle_2d_caption(x_lab="pseudo_step", y_lab="CD")
        == f"{bold('CD')} against {bold('pseudo_step')} for {bold('selected cases')}."
    )

    chart_items_in_row = Chart2D(x="total_forces/pseudo_step", y="total_forces/CD", items_in_row=2)

    assert (
        chart_items_in_row._handle_2d_caption(x_lab="pseudo_step", y_lab="CD")
        == f"{bold('CD')} against {bold('pseudo_step')}."
    )


def test_3d_caption_validity(cases):
    top_camera = Camera(
        position=(0, 0, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 1, 0),
        dimension=5,
        dimension_dir="width",
    )

    chart = Chart3D(
        section_title="Geometry",
        force_new_page=True,
        show="boundaries",
        camera=top_camera,
        fig_name="geo",
    )

    with pytest.raises(
        ValueError, match="List of captions and items_in_row cannot be used together."
    ):
        Chart3D(
            section_title="Geometry",
            force_new_page=True,
            show="boundaries",
            camera=top_camera,
            fig_name="geo",
            items_in_row=2,
            caption=["Caption 1", "Caption 2"],
        )

    with pytest.raises(
        ValueError, match="PatternCaption and items_in_row cannot be used together."
    ):
        Chart3D(
            section_title="Geometry",
            force_new_page=True,
            show="boundaries",
            camera=top_camera,
            fig_name="geo",
            items_in_row=2,
            caption=PatternCaption(pattern="This is case: [case.name] with ID: [case.id]"),
        )

    with pytest.raises(
        ValueError, match="Caption list is not the same length as the list of cases."
    ):
        chart.caption = ["Caption 1", "Caption 2", "Caption 3"]
        chart._check_caption_validity(cases)


def test_3d_caption(cases):
    top_camera = Camera(
        position=(0, 0, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 1, 0),
        dimension=5,
        dimension_dir="width",
    )

    chart = Chart3D(
        section_title="Geometry",
        force_new_page=True,
        show="boundaries",
        camera=top_camera,
        fig_name="geo",
    )

    chart.caption = "This is a caption."
    assert chart._handle_3d_caption() == "This is a caption."

    chart.caption = ["Caption 1", "Caption 2"]
    assert chart._handle_3d_caption(case_number=0) == "Caption 1"
    assert chart._handle_3d_caption(case_number=1) == "Caption 2"

    chart.caption = PatternCaption(pattern="This is case: [case.name] with ID: [case.id]")
    assert (
        chart._handle_3d_caption(case=cases[0])
        == "This is case: case-11111111-1111-1111-1111-111111111111-name with ID: case-11111111-1111-1111-1111-111111111111"
    )
    assert (
        chart._handle_3d_caption(case=cases[1])
        == "This is case: case-2222222222-2222-2222-2222-2222222222-name with ID: case-2222222222-2222-2222-2222-2222222222"
    )


@pytest.mark.usefixtures("mock_detect_latex_compiler")
def test_subfigure_row_splitting():
    report_doc = ReportDoc("tester")

    chart = Chart2D(
        x="nonlinear_residuals/pseudo_step",
        y="nonlinear_residuals/0_cont",
        section_title="Continuity convergence",
        fig_name="convergence_cont",
        items_in_row=2,
    )

    chart._add_row_figure(doc=report_doc.doc, img_list=["." for _ in range(6)], fig_caption="test")

    tex = report_doc.doc.dumps()

    lines = tex.split("\n")

    subplots_in_row = 0

    rows = 0

    in_subfigure = False
    in_figure = False

    caption_in_figure = False

    for line in lines:
        line = line.lstrip()
        if line.startswith(r"\caption") and in_figure and (not in_subfigure):
            caption_in_figure = True
        if line.startswith(r"\begin{subfigure}[t]"):
            in_subfigure = True
            subplots_in_row += 1
        if line.startswith(r"\end{subfigure}"):
            in_subfigure = False
        if line.startswith(r"\begin{figure}[h!]"):
            in_figure = True
        if line.startswith(r"\end{figure}"):
            assert subplots_in_row == 2
            subplots_in_row = 0
            rows += 1
            if rows == 3:
                assert caption_in_figure
            else:
                assert not caption_in_figure
            caption_in_figure = False
            in_figure = False


def test_multi_variable_chart_2d_one_case(cases, residual_plot_model):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    context = ReportContext(cases=[cases[0]])

    chart = Chart2D(
        x="nonlinear_residuals/pseudo_step",
        y=[f"nonlinear_residuals/{res}" for res in residuals_sa],
        section_title="Continuity convergence",
        fig_name="convergence_cont",
        separate_plots=True,
    )

    plot_model = chart.get_data([cases[0]], context)

    assert plot_model.x_data == residual_plot_model.x_data
    assert plot_model.y_data == residual_plot_model.y_data
    assert plot_model.x_label == residual_plot_model.x_label
    assert plot_model.y_label == "value"
    assert plot_model.legend == residuals_sa


def test_multi_variable_chart_2d_mult_cases(cases, two_var_two_cases_plot_model):
    loads = ["totalCL", "totalCD"]
    context = ReportContext(cases=cases)

    legend = []
    for case in cases:
        for load in loads:
            legend.append(f"{case.name} - {load}")

    chart = Chart2D(
        x="surface_forces/pseudo_step",
        y=[f"surface_forces/{load}" for load in loads],
        section_title="Loads convergence",
        fig_name="loads_conv",
        separate_plots=False,
    )

    plot_model = chart.get_data(cases, context)

    assert np.allclose(
        plot_model.x_data_as_np, two_var_two_cases_plot_model.x_data_as_np, rtol=1e-4, atol=1e-7
    )
    assert np.allclose(
        plot_model.y_data_as_np, two_var_two_cases_plot_model.y_data_as_np, rtol=1e-4, atol=1e-7
    )
    assert plot_model.x_label == two_var_two_cases_plot_model.x_label
    assert plot_model.y_label == two_var_two_cases_plot_model.y_label
    assert plot_model.legend == legend


def test_chart_2d_grid(cases):
    loads = ["totalCL", "totalCD"]
    context = ReportContext(cases=cases)

    chart = Chart2D(
        x="surface_forces/pseudo_step",
        y=[f"surface_forces/{load}" for load in loads],
        section_title="Loads convergence",
        fig_name="loads_conv",
        show_grid=True,
    )

    plot_model = chart.get_data(cases, context)
    fig = plot_model.get_plot()
    ax = fig.axes[0]

    assert all(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())


def test_residuals(cases, residual_plot_model):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    residuals = NonlinearResiduals()
    context = ReportContext(cases=[cases[0]])

    plot_model = residuals.get_data([cases[0]], context)

    assert plot_model.x_data == (np.array(residual_plot_model.x_data)[:, 1:]).tolist()
    assert plot_model.y_data == (np.array(residual_plot_model.y_data)[:, 1:]).tolist()
    assert plot_model.x_label == residual_plot_model.x_label
    assert plot_model.y_label == "residual values"
    assert plot_model.legend == residuals_sa
    # TODO: add case and test for residuals from SST
