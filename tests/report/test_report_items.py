import os

import pandas
import pytest
from pylatex import Document

from flow360 import Case, u
from flow360.component.case import CaseMeta
<<<<<<< HEAD
=======
from flow360.component.resource_base import local_metadata_builder
from flow360.component.utils import LocalResourceCache
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2
from flow360.exceptions import Flow360ValidationError
>>>>>>> 74d54440 (Fix validation error while using deprecated include/exclude Chart2D (#1029))
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.report_items import (
    Camera,
    Chart3D,
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

<<<<<<< HEAD
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
=======
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
            caption=["Caption 1", "Caption 2", "Caption 3"],
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
        chart.caption = ["Caption 1", "Caption 2", "Caption 3", "Caption 4"]
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


@check_figures_equal(extensions=["png"])
def test_plot_model_basic(fig_test, fig_ref):
    plot_model = PlotModel(
        x_data=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        y_data=[[4, 5, 6, 7, 8], [1, 2, 3, 4, 5]],
        x_label="argument",
        y_label="value",
        legend=["a", "b"],
    )

    original_subplots = plt.subplots

    def _fake_subplots(*args, **kwargs):
        ax = fig_test.subplots()
        return fig_test, ax

    plt.subplots = _fake_subplots

    try:
        fig = plot_model.get_plot()
    finally:
        plt.subplots = original_subplots

    # sanity: ensure it really did draw on fig_test
    assert fig is fig_test

    ax_ref = fig_ref.subplots()

    ax_ref.plot([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
    ax_ref.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    ax_ref.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(x, "g")))
    ax_ref.legend(["a", "b"])
    ax_ref.set_xlabel("argument")
    ax_ref.set_ylabel("value")
    ax_ref.grid(True)


@check_figures_equal(extensions=["png"])
def test_plot_model_secondary_x(fig_test, fig_ref):
    plot_model = PlotModel(
        x_data=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        y_data=[[4, 5, 6, 7, 8], [1, 2, 3, 4, 5]],
        secondary_x_data=[[0, 1, 1, 2, 2], [0, 1, 1, 2, 2]],
        secondary_x_label="arg2",
        x_label="argument",
        y_label="value",
        legend=["a", "b"],
    )

    original_subplots = plt.subplots

    def _fake_subplots(*args, **kwargs):
        ax = fig_test.subplots()
        return fig_test, ax

    plt.subplots = _fake_subplots

    try:
        fig = plot_model.get_plot()
    finally:
        plt.subplots = original_subplots

    # sanity: ensure it really did draw on fig_test
    assert fig is fig_test

    ax_ref = fig_ref.subplots()

    x1_changes = [1, 2, 4]
    x2 = [0, 1, 2]

    ax_ref.plot([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
    ax_ref.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    ax_ref.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(x, "g")))
    sec_ax = ax_ref.secondary_xaxis(location="top")
    sec_ax.set_xlabel("arg2")
    sec_ax.set_xticks(x1_changes, x2)
    ax_ref.legend(["a", "b"])
    ax_ref.set_xlabel("argument")
    ax_ref.set_ylabel("value")
    ax_ref.grid(True)


@check_figures_equal(extensions=["png"])
def test_plot_model_secondary_x_w_xlim(fig_test, fig_ref):
    plot_model = PlotModel(
        x_data=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        y_data=[[4, 5, 6, 7, 8], [1, 2, 3, 4, 5]],
        secondary_x_data=[[0, 1, 1, 2, 2], [0, 1, 1, 2, 2]],
        secondary_x_label="arg2",
        x_label="argument",
        y_label="value",
        legend=["a", "b"],
        xlim=(3, 5),
    )

    original_subplots = plt.subplots

    def _fake_subplots(*args, **kwargs):
        ax = fig_test.subplots()
        return fig_test, ax

    plt.subplots = _fake_subplots

    try:
        fig = plot_model.get_plot()
    finally:
        plt.subplots = original_subplots

    # sanity: ensure it really did draw on fig_test
    assert fig is fig_test

    ax_ref = fig_ref.subplots()

    x1_changes = [1, 2, 4]
    x2 = [0, 1, 2]

    ax_ref.plot([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
    ax_ref.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    ax_ref.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(x, "g")))
    sec_ax = ax_ref.secondary_xaxis(location="top")
    sec_ax.set_xlabel("arg2")
    sec_ax.set_xticks(x1_changes, x2)
    ax_ref.legend(["a", "b"])
    ax_ref.set_xlabel("argument")
    ax_ref.set_ylabel("value")
    ax_ref.grid(True)
    ax_ref.set_xlim(3, 5)


def test_multi_variable_chart_2d_one_case(cases, residual_plot_model_SA):
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

    assert plot_model.x_data == residual_plot_model_SA.x_data
    assert plot_model.y_data == residual_plot_model_SA.y_data
    assert plot_model.x_label == residual_plot_model_SA.x_label
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


def test_residuals_same(cases, residual_plot_model_SA, residual_plot_model_SST):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    residuals_sst = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_k", "6_omega"]

    residuals = NonlinearResiduals()
    context = ReportContext(cases=[cases[0]])

    plot_model_SA = residuals.get_data([cases[0]], context)

    plot_model_SST = residuals.get_data([cases[2]], context)

    plot_model_both = residuals.get_data([cases[0], cases[2]], context)

    assert plot_model_SA.x_data == (np.array(residual_plot_model_SA.x_data)[:, 1:]).tolist()
    assert plot_model_SA.y_data == (np.array(residual_plot_model_SA.y_data)[:, 1:]).tolist()
    assert plot_model_SA.x_label == residual_plot_model_SA.x_label
    assert plot_model_SA.y_label == "residual values"
    assert plot_model_SA.legend == residuals_sa

    assert plot_model_SST.x_data == (np.array(residual_plot_model_SST.x_data)[:, 1:]).tolist()
    assert plot_model_SST.y_data == (np.array(residual_plot_model_SST.y_data)[:, 1:]).tolist()
    assert plot_model_SST.x_label == residual_plot_model_SST.x_label
    assert plot_model_SST.y_label == "residual values"
    assert plot_model_SST.legend == residuals_sst

    assert plot_model_both.x_data == (
        (np.array(residual_plot_model_SA.x_data)[:, 1:]).tolist()
        + (np.array(residual_plot_model_SST.x_data)[:, 1:]).tolist()
    )
    assert plot_model_both.y_data == (
        (np.array(residual_plot_model_SA.y_data)[:, 1:]).tolist()
        + (np.array(residual_plot_model_SST.y_data)[:, 1:]).tolist()
    )
    assert plot_model_both.x_label == residual_plot_model_SST.x_label
    assert plot_model_both.y_label == "residual values"


def test_multiple_point_variables_on_chart2d(cases, here):
    loads = ["CL", "CD"]
    context = ReportContext(cases=cases[:2])
    chart = Chart2D(
        x="params/operating_condition/beta",
        y=[f"total_forces/averages/{load}" for load in loads],
        section_title="Loads on beta",
        fig_name="loads_beta",
        show_grid=True,
    )

    plot_model = chart.get_data(cases=cases[:2], context=context)

    ys_to_plot = np.zeros((len(loads), 2))
    xs_to_plot = np.zeros((len(loads), 2))

    for idx0, case in enumerate(cases[:2]):
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.id, "results", "total_forces_v2.csv"),
            skipinitialspace=True,
        )
        to_avg = round(len(load_data) * 0.1)
        xs_to_plot[:, idx0] = case.params.operating_condition.beta.value
        for idx1, load in enumerate(loads):
            ys_to_plot[idx1, idx0] = np.average(load_data[load].iloc[-to_avg:])

    assert np.allclose(plot_model.x_data_as_np, xs_to_plot)
    assert np.allclose(plot_model.y_data_as_np, ys_to_plot)
    assert plot_model.x_label == "beta [degree]"
    assert plot_model.y_label == "value"
    assert plot_model.legend == loads


def test_dataitem_point_variables_on_chart2d(cases, here):
    loads_surf = ["totalCFy", "totalCFx"]
    loads = ["CFy", "CFx"]

    dataitems = [
        DataItem(data=f"surface_forces/{load}", operations=[Average(fraction=0.2)])
        for load in loads_surf
    ]
    context = ReportContext(cases=cases[:2])
    chart = Chart2D(
        x="params/operating_condition/beta",
        y=dataitems,
        section_title="Loads on beta",
        fig_name="loads_beta",
        show_grid=True,
    )

    plot_model = chart.get_data(cases=cases[:2], context=context)

    ys_to_plot = np.empty((len(loads), 2))
    xs_to_plot = np.empty((len(loads), 2))

    for idx0, case in enumerate(cases[:2]):
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.id, "results", "total_forces_v2.csv"),
            skipinitialspace=True,
        )
        to_avg = round(len(load_data) * 0.2)
        xs_to_plot[:, idx0] = case.params.operating_condition.beta.value
        for idx1, load in enumerate(loads):
            ys_to_plot[idx1, idx0] = np.average(load_data[load].iloc[-to_avg:])

    assert np.allclose(plot_model.x_data_as_np, xs_to_plot)
    assert np.allclose(plot_model.y_data_as_np, ys_to_plot)
    assert plot_model.x_label == "beta [degree]"
    assert plot_model.y_label == "value"
    assert plot_model.legend == loads_surf


def test_dataitem_result_csv_compatibility(cases, here):
    loads = ["CFy", "CFx"]

    dataitems = [
        DataItem(data=f"total_forces/{load}", operations=[Average(fraction=0.2)]) for load in loads
    ]
    context = ReportContext(cases=cases[:2])
    chart = Chart2D(
        x="params/operating_condition/beta",
        y=dataitems,
        section_title="Loads on beta",
        fig_name="loads_beta",
        show_grid=True,
    )

    plot_model = chart.get_data(cases=cases[:2], context=context)

    ys_to_plot = np.empty((len(loads), 2))
    xs_to_plot = np.empty((len(loads), 2))

    for idx0, case in enumerate(cases[:2]):
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.id, "results", "total_forces_v2.csv"),
            skipinitialspace=True,
        )
        to_avg = round(len(load_data) * 0.2)
        xs_to_plot[:, idx0] = case.params.operating_condition.beta.value
        for idx1, load in enumerate(loads):
            ys_to_plot[idx1, idx0] = np.average(load_data[load].iloc[-to_avg:])

    assert np.allclose(plot_model.x_data_as_np, xs_to_plot)
    assert np.allclose(plot_model.y_data_as_np, ys_to_plot)
    assert plot_model.x_label == "beta [degree]"
    assert plot_model.y_label == "value"
    assert plot_model.legend == loads


@pytest.mark.filterwarnings("ignore:The `__fields__` attribute is deprecated")
def test_transient_forces(here, cases_transient):
    loads = ["CFx", "CFy"]
    case_id = "case-444444444-444444-4444444444-44444444"

    context = ReportContext(cases=[cases_transient[0]])

    # expected data
    data = pd.read_csv(
        os.path.join(here, "..", "data", case_id, "results", "total_forces_v2.csv"),
        skipinitialspace=True,
    )

    data["cumulative_pseudo_step"] = get_cumulative_pseudo_time_step(data["pseudo_step"])

    data["time"] = data["physical_step"] * 0.1

    loads_by_physical_step = [
        get_last_time_step_values(data["pseudo_step"], data[load]) for load in loads
    ]

    chart_forces_pseudo = Chart2D(
        x="total_forces/pseudo_step",
        y=[f"total_forces/{load}" for load in loads],
        section_title="Loads pseudo",
        fig_name="loads_pseudo",
    )

    chart_forces_physical = Chart2D(
        x="total_forces/physical_step",
        y=[f"total_forces/{load}" for load in loads],
        section_title="Loads physical",
        fig_name="loads_physical",
    )

    chart_forces_time = Chart2D(
        x="total_forces/time",
        y=[f"total_forces/{load}" for load in loads],
        section_title="Loads time",
        fig_name="loads_time",
    )

    plot_model_pseudo = chart_forces_pseudo.get_data([cases_transient[0]], context)
    plot_model_physical = chart_forces_physical.get_data([cases_transient[0]], context)
    plot_model_time = chart_forces_time.get_data([cases_transient[0]], context)

    assert plot_model_pseudo.x_data == [data["cumulative_pseudo_step"].to_list()] * len(loads)
    assert plot_model_pseudo.y_data == [data[load].to_list() for load in loads]

    assert plot_model_physical.x_data == [
        get_last_time_step_values(data["pseudo_step"], data["physical_step"])
    ] * len(loads)
    assert plot_model_physical.y_data == loads_by_physical_step

    assert plot_model_time.x_data == [
        get_last_time_step_values(data["pseudo_step"], data["time"])
    ] * len(loads)
    assert plot_model_time.y_data == loads_by_physical_step


def test_transient_residuals_pseudo(here, cases_transient):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    case_id = "case-444444444-444444-4444444444-44444444"

    context = ReportContext(cases=[cases_transient[0]])

    # expected data
    data = pd.read_csv(
        os.path.join(here, "..", "data", case_id, "results", "nonlinear_residual_v2.csv"),
        skipinitialspace=True,
    )

    cum_ts = get_cumulative_pseudo_time_step(data["pseudo_step"])
    data["cumulative_pseudo_step"] = cum_ts

    residuals = NonlinearResiduals()

    plot_model_residuals = residuals.get_data(cases=[cases_transient[0]], context=context)

    assert plot_model_residuals.x_data == [(data["cumulative_pseudo_step"][1:]).to_list()] * len(
        residuals_sa
    )
    assert plot_model_residuals.y_data == [(data[res][1:]).to_list() for res in residuals_sa]
    assert plot_model_residuals.secondary_x_data is None

    residuals = NonlinearResiduals(xlim=ManualLimit(lower=200, upper=380))

    plot_model_residuals = residuals.get_data(cases=[cases_transient[0]], context=context)

    assert np.allclose(
        plot_model_residuals.secondary_x_data_as_np,
        np.array([data["physical_step"][1:].to_numpy()] * len(residuals_sa)),
    )


def test_include_exclude(here, cases):
    chart = Chart2D(
        x="surface_forces/averages/totalCD",
        y="surface_forces/averages/totalCL",
        section_title="CL/CD",
        fig_name="clcd",
        include=["blk-1/BODY"],
    )

    context = ReportContext(cases=cases[:2])

    plot_model = chart.get_data(cases=cases[:2], context=context)

    expected_xs = []
    expected_ys = []

    # expected data
    for idx0, case in enumerate(cases[:2]):
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.id, "results", "surface_forces_v2.csv"),
            skipinitialspace=True,
        )
        to_avg = round(len(load_data) * 0.1)

        expected_xs.append(np.average(load_data["blk-1/BODY_CD"].iloc[-to_avg:]))
        expected_ys.append(np.average(load_data["blk-1/BODY_CL"].iloc[-to_avg:]))

    assert np.allclose(np.array(expected_xs), plot_model.x_data_as_np)
    assert np.allclose(np.array(expected_ys), plot_model.y_data_as_np)

    chart = Chart2D(
        x="total_forces/averages/CD",
        y="total_forces/averages/CL",
        section_title="CL/CD",
        fig_name="clcd",
        include=["blk-1/BODY"],
    )

    with pytest.raises(AttributeError):
        plot_model = chart.get_data(cases=cases[:2], context=context)

    with pytest.raises(Flow360ValidationError):
        chart = Chart2D(
            x=Delta(data="surface_forces/averages/totalCD"),
            y="surface_forces/averages/totalCL",
            section_title="CL/CD",
            fig_name="clcd",
            include=["blk-1/BODY"],
        )


def test_in_path_averages(here, cases):
    dataitem = DataItem(
        data="total_forces/averages/CL",
        operations=[Expression(expr="CL*beta")],
        variables=[Variable(name="beta", data="params/operating_condition/beta")],
    )

    assert dataitem.operations[0] == Expression(expr="CL*beta")
    assert dataitem.operations[1] == Average(fraction=0.1)

    cl_beta = dataitem.calculate(case=cases[1], cases=cases)

    load_data = pd.read_csv(
        os.path.join(here, "..", "data", cases[1].id, "results", "total_forces_v2.csv"),
        skipinitialspace=True,
    )
    to_avg = round(len(load_data) * 0.1)

    cl_beta_expected = (
        np.average(load_data["CL"].iloc[-to_avg:]) * cases[1].params.operating_condition.beta.value
    )

    assert dataitem.operations[2] == Average(fraction=0.1)
    assert dataitem.operations[1] == Expression(expr="CL*beta")

    assert np.allclose(cl_beta, cl_beta_expected)
>>>>>>> 74d54440 (Fix validation error while using deprecated include/exclude Chart2D (#1029))
