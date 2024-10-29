import os
from flow360.plugins.report.report import Report
from flow360.plugins.report.report_items import Summary, Inputs, Table, Chart2D, Chart3D
from flow360.plugins.report.utils import Delta


import flow360 as fl
from flow360 import log

log.set_logging_level("DEBUG")
fl.UserConfig.set_profile("auto_test_1")
fl.Env.dev.active()

here = os.path.dirname(os.path.abspath(__file__))


case1 = fl.Case("case-21469d5e-257d-49de-8f5d-97f27c526a47")  # alpha=5
case2 = fl.Case("case-8f1e31fc-e8df-408f-aab8-62507bf85bf5")  # alpha=10
case3 = fl.Case("case-73e1d12f-a8d1-477c-95cf-45f6685e7971")  # alpha=15

SOLVER_VERSION = "reportPipeline-24.10.2"


report = Report(
    items=[
        Summary(text="Analysis of a new feature"),
        Inputs(),
        Table(
            data_path=[
                "params/reference_geometry/area",
                "total_forces/averages/CD",
                Delta(data_path="total_forces/averages/CD", ref_index=0),
            ],
            section_title="My Favourite Quantities",
        ),
        Chart3D(
            section_title="Chart3D Testing",
            fig_size=0.4,
            fig_name="c3d_std",
            force_new_page=True,
            show="boundaries",
        ),
        Chart3D(
            section_title="Chart3D Rows Testing",
            items_in_row=-1,
            fig_name="c3d_rows",
            show="boundaries",
            field="yPlus",
            limits=(0.1, 82),
        ),
        Chart3D(
            section_title="Q-criterion in row",
            items_in_row=-1,
            fig_name="c3d_rows_qcriterion",
            show="qcriterion",
            field="Mach",
            limits=(0, 0.346),
        ),
        Chart2D(
            x=["total_forces/pseudo_step", "total_forces/pseudo_step"],
            section_title="Sanity Check Step against Step",
            fig_name="step_fig",
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CL"],
            section_title="Global Coefficient of Lift (just first Case)",
            fig_name="cl_fig",
            select_indices=[1],
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CFy"],
            section_title="Global Coefficient of Force in Y (subfigure and combined)",
            fig_name="cd_fig",
            items_in_row=-1,
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CFy"],
            section_title=None,
            fig_name="cd_comb_fig",
            single_plot=True,
        ),
        Chart2D(
            data_path=["nonlinear_residuals/pseudo_step", "nonlinear_residuals/1_momx"],
            section_title=None,
            fig_name="residuals",
            single_plot=True,
        ),
    ],
    include_case_by_case=True,
)


report = report.create_in_cloud(
    "running_report_from_python", [case1, case2, case3], solver_version=SOLVER_VERSION
)

report.download("report.pdf")
