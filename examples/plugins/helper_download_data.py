import os
import json
from plugins.report.report import Report
from plugins.report.report_items import Summary, Inputs, Table, Chart2D, Chart3D
from plugins.report.utils import Delta


import flow360 as fl

here = os.path.dirname(os.path.abspath(__file__))


fl.UserConfig.set_profile("auto_test_1")


case1 = fl.Case("case-a8c58253-d76c-498c-8827-4a1dc3772389")  # alpha=5
case2 = fl.Case("case-949b8362-feb5-4c9d-92f0-1d551f1d5f05")  # alpha=10
case3 = fl.Case("case-7b3233b4-eaf2-4724-9b8c-926b9807049a")  # alpha=15
cases = [case1, case2, case3]


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
            data_path=["nonlinear_residuals/pseudo_step", "nonlinear_residuals/momx"],
            section_title=None,
            fig_name="cd_comb_fig",
            background=None,
            single_plot=True,
        ),
    ],
    include_case_by_case=True,
)


print(report.get_requirements())


for requirement in report.get_requirements():
    for case in cases:
        if os.path.basename(requirement) == 'manifest.json':
            continue
        case._download_file(file_name=requirement, to_folder=os.path.join(here, case.name))