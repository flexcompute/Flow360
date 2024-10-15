import os
from flow360.plugins.report.report import Report
from flow360.plugins.report.report_items import Summary, Inputs, Table, Chart2D, Chart3D
from flow360.plugins.report.utils import Delta


import flow360 as fl
from flow360 import log

log.set_logging_level("DEBUG")

here = os.path.dirname(os.path.abspath(__file__))


case1 = fl.Case("case-a8c58253-d76c-498c-8827-4a1dc3772389")  # alpha=5
case2 = fl.Case("case-949b8362-feb5-4c9d-92f0-1d551f1d5f05")  # alpha=10
case3 = fl.Case("case-7b3233b4-eaf2-4724-9b8c-926b9807049a")  # alpha=15


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
        ),
        Chart3D(section_title="Chart3D Rows Testing", items_in_row=-1, fig_name="c3d_rows"),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/pseudo_step"],
            section_title="Sanity Check Step against Step",
            fig_name="step_fig",
            background=None,
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CL"],
            section_title="Global Coefficient of Lift (just first Case)",
            fig_name="cl_fig",
            background=None,
            select_indices=[1],
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CFy"],
            section_title="Global Coefficient of Force in Y (subfigure and combined)",
            fig_name="cd_fig",
            background=None,
            items_in_row=-1,
        ),
        Chart2D(
            data_path=["total_forces/pseudo_step", "total_forces/CFy"],
            section_title=None,
            fig_name="cd_comb_fig",
            background=None,
            single_plot=True,
        ),
        Chart2D(
            data_path=["nonlinear_residuals/pseudo_step", "nonlinear_residuals/1_momx"],
            section_title=None,
            fig_name="residuals",
            background=None,
            single_plot=True,
        ),
    ],
    include_case_by_case=True,
)


fl.Env.dev.active()

report_wait = report.create_in_cloud("running_report_from_python", [case1, case2, case3], solver_version='report-24.10.0')


print(report_wait)