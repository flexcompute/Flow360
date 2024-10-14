import os
import pytest
import tempfile
from pylatex import Document, Section, Figure
from report import Report, Chart2D, Chart3D, Table, Delta, Summary, Inputs, get_case_from_id

from flow360 import Case

# Questions
# How best to emulate Chart3D image download?
# Is it worth loading the generated pdf into python and asserting quality?

# Something may be causing current directory to not be reset after running test - failing in VSCode but works when running pytest via CLI

@pytest.mark.usefixtures("s3_download_override")
def test_chart2d(mock_id, mock_response):
    """Test Chart2D is creating png files from mock case."""

    case = Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, destination=temp_dir)
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))

        os.chdir(temp_dir)
        test_doc = Document()

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

        test_doc = Document()
        Chart2D(
            data_path=data_path,
            section_title=section_title,
            fig_name="chart2d_test1_",
            items_in_row=2
        ).get_doc_item([case, case], test_doc, Section)

        # Assert pngs exist, section title is right, image plot is the correct size for 2 in row
        assert "chart2d_test1_om6wing-from-yaml.png" in os.listdir()
        assert section_title in str(test_doc.data[1])
        assert "0.49\\textwidth" in str(test_doc.data[4].arguments)

        test_doc = Document()
        Chart2D(
            data_path=data_path,
            section_title=section_title,
            fig_name="chart2d_test2_",
            items_in_row=None
        ).get_doc_item([case, case], test_doc, Section)

        # Assert pngs exist, section title is right, image plot is the correct size for 2 in row
        assert "chart2d_test2_om6wing-from-yaml.png" in os.listdir()
        assert isinstance(test_doc.data[2], Figure) and isinstance(test_doc.data[3], Figure)

def test_chart3d():
    # To be written once pipeline design has been finalised
    # Chart3D(
    #     section_title="Chart3D Testing",
    #     fig_size=0.4,
    #     fig_name="c3d_std",
    # )

    # Plan for mock server request:
    # Ideally monkeypatch _get_image with function that returns png string
    # This may fail to pass through asyncio.gather
    # Otherwise monkeypatch process_3d_images and just copy mock png file to target dir
    # Only tests latex code for placing image, which is functionally similar to Chart2D
    pass

@pytest.mark.usefixtures("s3_download_override")
def test_delta(mock_id, mock_response):
    """Test Delta is calculating a difference between two cases."""
    
    case = Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, destination=temp_dir)
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))

    # Test get_case_from_id
    ref_case = get_case_from_id(case.id, [case, case])

    zero_case = Delta(data_path="total_forces/CD", ref_case_id=case.id).calculate(case, ref_case)
    assert zero_case == 0

@pytest.mark.usefixtures("s3_download_override")
def test_table(mock_id, mock_response):
    """Test Table is creating a latex table with the right shape and content."""

    case = Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, destination=temp_dir)
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))

    test_doc = Document()
    Table(
        data_path=[
            "params_as_dict/geometry/refArea",
            Delta(data_path="total_forces/CD", ref_case_id=case.id)
        ],
        section_title="Test Table",
        custom_headings=None
    ).get_doc_item([case, case], test_doc, Section)

    # Tests that the number of rows and hline commands
    # Set reduces length to 1 hline, 1 rowcolor and 3 unique rows
    assert len(test_doc.data[2].data) == 8 and len(set(test_doc.data[2].data)) == 5

    # Test custom headings
    test_doc = Document()
    Table(
        data_path=[
            "params_as_dict/geometry/refArea", 
            Delta(data_path="total_forces/CD", ref_case_id=case.id)
        ],
        section_title="Test Table",
        custom_headings=["\textit{AnItalicHeading}", "1.£,$;%:^#&?*()[]{}"]
    ).get_doc_item([case, case], test_doc, Section)

    # \textit doesn't work but check is to see if it throws an error
    # pylatex escapes all these characters automatically so checking that it does it properly
    assert r"1.£,\$;\%:\^{}\#\&?*(){[}{]}\{\}}\\" in str(test_doc.data[2].data[2])

@pytest.mark.usefixtures("s3_download_override")
def test_create_report(mock_id, mock_response):
    """Test report creation completes successfully."""

    # This does fail if pylatex throws an error - even if a mostly good pdf is produced
    # Just doing case_by_case True as it calls all the False side as well
    case = Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, destination=temp_dir)
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))

        report = Report(
            items=[
                Summary(text='Report Summary Test'),
                # Inputs(), # Currently not included as waiting on data selection

                Table(
                    data_path=[
                        "params_as_dict/geometry/refArea",
                        Delta(data_path="total_forces/CD", ref_case_id=case.id),
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
            include_case_by_case=True,
            output_dir=temp_dir
        )

        report.create_pdf("test_report", [case, case])