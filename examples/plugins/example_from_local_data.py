import os
from flow360.plugins.report.report import Report, DataNode
from flow360.plugins.report.report_items import Summary, Inputs, Table, Chart2D, Chart3D
from flow360.plugins.report.utils import Delta


import flow360 as fl
from flow360 import log

log.set_logging_level("DEBUG")
fl.Env.dev.active()
fl.UserConfig.set_profile("auto_test_1")


here = os.path.dirname(os.path.abspath(__file__))

# print(fl.Case("case-21469d5e-257d-49de-8f5d-97f27c526a47").info.user_id)


data_path = DataNode()


case1 = fl.Case.from_local_storage(
    "case-21469d5e-257d-49de-8f5d-97f27c526a47",
    "Case_f_alpha=5",
    os.path.join(here, "Case_f_alpha=5"),
    user_id="AIDAU77I6BZ2QYZLLVSRW",
)
case2 = fl.Case.from_local_storage(
    "case-8f1e31fc-e8df-408f-aab8-62507bf85bf5",
    "Case_f_alpha=10",
    os.path.join(here, "Case_f_alpha=10"),
    user_id="AIDAU77I6BZ2QYZLLVSRW",
)
case3 = fl.Case.from_local_storage(
    "case-73e1d12f-a8d1-477c-95cf-45f6685e7971",
    "Case_f_alpha=15",
    os.path.join(here, "Case_f_alpha=15"),
    user_id="AIDAU77I6BZ2QYZLLVSRW",
)


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

# # NOTE: There's a bug where something is being cached between create_pdf calls like this
# # The issue seems to affect _assemble_fig_rows
# # report.create_pdf("test_report_landscape", [a2_case, b2_case, other_a2_case], landscape=True)
# report.create_pdf("test_report_portrait", [case1, case2, case3], landscape=True, data_storage=os.path.join(here, 'my_report'))


report_filename = os.path.join(here, "my_report", "report.json")
with open(report_filename, "w") as f:
    f.write(report.model_dump_json())


Report(filename=report_filename).create_pdf(
    "test_report_portrait",
    [case1, case2, case3],
    landscape=True,
    data_storage=os.path.join(here, "my_report"),
)
