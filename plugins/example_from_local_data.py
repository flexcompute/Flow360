import os
from plugins.report.report import Report
from plugins.report.report_items import Summary, Inputs, Table, Chart2D, Chart3D
from plugins.report.utils import Delta


import flow360 as fl
from flow360 import log

log.set_logging_level("DEBUG")

here = os.path.dirname(os.path.abspath(__file__))


case1 = fl.Case.from_local_storage(
    "case-a8c58253-d76c-498c-8827-4a1dc3772389", "Case_f_alpha=5", os.path.join(here, "Case_f_alpha=5")
)
case2 = fl.Case.from_local_storage(
    "case-949b8362-feb5-4c9d-92f0-1d551f1d5f05", "Case_f_alpha=10", os.path.join(here, "Case_f_alpha=10")
)
case3 = fl.Case.from_local_storage(
    "case-7b3233b4-eaf2-4724-9b8c-926b9807049a", "Case_f_alpha=15", os.path.join(here, "Case_f_alpha=15")
)



print(case1.results.total_forces.averages['CD'])
print(case2.results.total_forces.averages['CD'])
print(case3.results.total_forces.averages['CD'])

# report = Report(
#     items=[
#         Table(
#             data_path=[
#                 "params/reference_geometry/area",
#                 "total_forces/averages/CD",
#                 Delta(data_path="total_forces/averages/CD", ref_index=0),
#             ],
#             section_title="My Favourite Quantities",
#         )
#     ]
# )

# report.create_pdf("test_report_portrait", [case1, case2, case3], landscape=True, data_storage=os.path.join(here, 'my_report'))


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
    ],
    include_case_by_case=True,
)

# # NOTE: There's a bug where something is being cached between create_pdf calls like this
# # The issue seems to affect _assemble_fig_rows
# # report.create_pdf("test_report_landscape", [a2_case, b2_case, other_a2_case], landscape=True)
report.create_pdf("test_report_portrait", [case1, case2, case3], landscape=True, data_storage=os.path.join(here, 'my_report'))
