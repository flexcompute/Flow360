import os

import pandas
import pytest
from pylatex import Document

from flow360 import Case, u
from flow360.component.case import CaseMeta
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.report_items import (
    Camera,
    Chart2D,
    Chart3D,
    FixedRangeLimit,
    ManualLimit,
    SubsetLimit,
    Table,
    human_readable_formatter,
)
from flow360.plugins.report.utils import Average, DataItem, Delta, Expression


@pytest.fixture
def here():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def cases(here):
    case_ids = [
        "case-11111111-1111-1111-1111-111111111111",
        "case-2222222222-2222-2222-2222-2222222222",
    ]
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
    return cases


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

    exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]

    avg = Average(fraction=0.1)
    CD = DataItem(data="surface_forces/totalCD", exclude=exclude, title="CD", operations=avg)

    CL = DataItem(data="surface_forces/totalCL", exclude=exclude, title="CL", operations=avg)

    CLCompare = DataItem(
        data="surface_forces",
        exclude=exclude,
        title="CL_compare",
        operations=[Expression(expr="totalCL - 1.5467"), avg],
    )

    CDCompare = DataItem(
        data="surface_forces",
        exclude=exclude,
        title="CD_compare",
        operations=[Expression(expr="totalCD - 0.02100"), avg],
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
    }
    df_expected = pandas.DataFrame(expected_data)
    df_expected["Case No."] = df_expected["Case No."].astype("Int64")
    print(df_expected)

    pandas.testing.assert_frame_equal(table_df, df_expected)


def test_calculate_y_lim(cases, here):
    chart = Chart2D(
        x="total_forces/pseudo_step",
        y="total_forces/CD",
    )
    case = cases[0]
    case_data = pandas.read_csv(
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


def test_2d_caption(cases):
    chart = Chart2D(
        x="total_forces/pseudo_step",
        y="total_forces/CD",
    )

    chart.caption = "This is a caption."
    chart._handle_2d_caption()


def test_3d_caption(cases):

    top_camera = Camera(
        position=(0, 0, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 1, 0),
        dimension=5,
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
    back_camera = Camera(position=(1, 0, 0), up=(0, 0, 1), dimension=2.5, dimension_dir="width")
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

    cameras_geo = [
        top_camera,
        side_camera,
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

    # work in progress
