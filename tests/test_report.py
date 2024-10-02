import os
import pytest
from pylatex import Document, Section, Figure
from report import Report, Chart2D, Chart3D, Table, Delta, Summary, Inputs, get_case_from_id

from flow360 import Case, SI_unit_system, Flow360Params, Geometry, FreestreamFromVelocity, air

# Questions
# What can I use as a preloaded Case with plotable data?
# How best to emulate Chart3D image download?
# Is it worth loading the generated pdf into python and asserting quality?
# Is creating a temp folder ok?


a2_case = Case("9d86fc07-3d43-4c72-b324-7acad033edde")
b2_case = Case("bd63add6-4093-4fca-95e8-f1ff754cfcd9")
cases = [a2_case, b2_case]

# with SI_unit_system:
#     case1 = Case.create(
#         name="hi",
#         params=Flow360Params(
#             geometry=Geometry(mesh_unit="m"),
#             freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
#             fluid_properties=air,
#             boundaries={},
#         ),
#         volume_mesh_id="00000000-0000-0000-0000-000000000000"
#     )

def _reset_test():
    for filename in os.listdir():
        if ".png" in filename or "test_report" in filename:
            os.remove(filename)

    return Document()
    
def test_chart2d():
    os.chdir("tests/temp")
    test_doc = _reset_test()

    data_path=["total_forces/pseudo_step", "total_forces/CFy"]
    section_title="Test Title"
    fig_name="chart2d_test_fig"

    # These should fail due to args:
    # items_in_row cannot be 1
    with pytest.raises(ValueError):
        Chart2D(
            data_path=data_path,
            section_title=section_title,
            fig_name=fig_name,
            items_in_row=1
        )

    # items_in_row and single_plot cannot be set together
    with pytest.raises(ValueError):
        Chart2D(
            data_path=data_path,
            section_title=section_title,
            fig_name=fig_name,
            items_in_row=2,
            single_plot=True
        )

    test_doc = _reset_test()
    Chart2D(
        data_path=data_path,
        section_title=section_title,
        fig_name=fig_name,
        items_in_row=2
    ).get_doc_item([a2_case, b2_case], test_doc, Section)

    # Assert pngs exist, section title is right, image plot is the correct size for 2 in row
    assert len(os.listdir()) == 2
    assert section_title in str(test_doc.data[1])
    assert "0.49\\textwidth" in str(test_doc.data[2].data[0])

    test_doc = _reset_test()
    Chart2D(
        data_path=data_path,
        section_title=section_title,
        fig_name=fig_name,
        items_in_row=None
    ).get_doc_item(cases, test_doc, Section)

    # Assert pngs exist, section title is right, image plot is the correct size for 2 in row
    assert len(os.listdir()) == 2
    assert isinstance(test_doc.data[2], Figure) and isinstance(test_doc.data[3], Figure)

def test_chart3d():
    # To be written once real pipeline is alive
    Chart3D
    pass

def test_delta():
    # Test get_case_from_id
    ref_case = get_case_from_id(a2_case.id, [a2_case, b2_case])

    zero_case = Delta(data_path="total_forces/CD", ref_case_id=a2_case.id).calculate(a2_case, ref_case)
    assert zero_case == 0
    
    b2_case_delta = Delta(data_path="total_forces/CD", ref_case_id=a2_case.id).calculate(b2_case, ref_case)
    assert round(b2_case_delta, 3) == -0.318

def test_table():

    test_doc = Document()
    Table(
        data_path=[
            "params_as_dict/geometry/refArea", 
            Delta(data_path="total_forces/CD", ref_case_id=a2_case.id)
        ],
        section_title="Test Table",
        custom_headings=None
    ).get_doc_item(cases, test_doc, Section)

    # Tests that the number of rows and hline commands
    # Set reduces length to 1 hline and 3 unique rows
    assert len(test_doc.data[2].data) == 7 and len(set(test_doc.data[2].data)) == 4

    # Test custom headings
    test_doc = Document()
    Table(
        data_path=[
            "params_as_dict/geometry/refArea", 
            Delta(data_path="total_forces/CD", ref_case_id=a2_case.id)
        ],
        section_title="Test Table",
        custom_headings=["\textit{AnItalicHeading}", "1.£,$;%:^#&?*()[]{}"]
    ).get_doc_item(cases, test_doc, Section)

    # \textit doesn't work but check is to see if it throws an error
    # pylatex escapes all these characters automatically so checking that it does it properly
    assert r"1.£,\$;\%:\^{}\#\&?*(){[}{]}\{\}}\\" in str(test_doc.data[2].data[1])

    # Just checking that neither throw errors - both are wrappers for Table so shouldn't
    Summary(text="Test Summary Text.")
    Inputs()

def test_report():
    # This does fail if pylatex throws an error - even if a mostly good pdf is produced
    # Just doing case_by_case True as it calls all the False side as well
    os.chdir(os.path.dirname(__file__) + "/temp")
    _reset_test()
    report = Report(
        items=[
            Summary(text='Report Summary Test'),
            Inputs(),

            Table(
                data_path=[
                    "params_as_dict/geometry/refArea", 
                    Delta(data_path="total_forces/CD", ref_case_id=a2_case.id),
                ],
                section_title="My Favourite Quantities",
            ),
            # Chart3D(
            #     section_title="Chart3D Testing",
            #     fig_size=0.5,
            #     fig_name="Unused"
            # ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/pseudo_step"],
                section_title="Sanity Check Step against Step",
                fig_name="step_fig",
                background=None,
            ),
        ],
        include_case_by_case=True
    )

    report.create_pdf("test_report", cases)