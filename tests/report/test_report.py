import os
import re

import pytest

from flow360 import Case
from flow360.component.case import CaseMeta
from flow360.plugins.report.report import Report, ReportDraft, ReportTemplate
from flow360.plugins.report.report_doc import ReportDoc
from flow360.plugins.report.report_items import Chart2D, Inputs, Summary, Table
from flow360.plugins.report.utils import RequirementItem


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


def test_report_init():
    rpt = Report("report-idxyz342-dasdad-dsadasda-3fsfdsf")
    assert rpt.id == "report-idxyz342-dasdad-dsadasda-3fsfdsf"


def test_reportapi_submit(mocker):
    mock_post = mocker.patch.object(
        ReportDraft._webapi, "post", return_value={"id": "report-idxyz342-dasdad-dsadasda-3fsfdsf"}
    )
    response = ReportDraft.submit(
        name="My Report", case_ids=["case1", "case2"], config="{}", solver_version="release-1.2.3"
    )
    mock_post.assert_called_once()
    assert isinstance(response, Report)
    assert response.id == "report-idxyz342-dasdad-dsadasda-3fsfdsf"


def test_reporttemplate_init_validation():
    with pytest.raises(ValueError, match="Duplicate fig_name"):
        ReportTemplate(
            title="Test Report",
            items=[
                Summary(text="Test summary"),
                Chart2D(
                    x="params/operating_condition/velocity_magnitude",
                    y="total_forces/CL",
                    fig_name="myfig",
                ),
                Chart2D(
                    x="params/time_stepping/type_name",
                    y="total_forces/CD",
                    fig_name="myfig",  # duplicate name
                ),
            ],
        )

    template = ReportTemplate(
        title="Another Report",
        items=[Summary(), Inputs(), Chart2D(x="params/version", y="total_forces/CD")],
    )
    assert len(template.items) == 3


def test_reporttemplate_create_in_cloud(mocker, cases):
    mock_submit = mocker.patch.object(ReportDraft, "submit", return_value="mock-response")
    template = ReportTemplate(title="Cloud Report", items=[Summary(), Inputs()])
    resp = template.create_in_cloud(name="CloudTest", cases=cases, solver_version="release-1.2.3")
    mock_submit.assert_called_once()

    call_args, call_kwargs = mock_submit.call_args
    assert call_kwargs["name"] == "CloudTest"
    assert call_kwargs["case_ids"] == [c.id for c in cases]
    assert call_kwargs["config"] is not None
    assert call_kwargs["solver_version"] == "release-1.2.3"
    assert resp == "mock-response"


@pytest.mark.usefixtures("generate_pdf", "mock_detect_latex_compiler")
def test_reporttemplate_create_pdf(cases):
    template = ReportTemplate(
        title="PDF Test", items=[Summary(), Inputs()], include_case_by_case=True
    )
    template.create_pdf("test_report.pdf", cases, data_storage=".")
    template.create_pdf("test_report", cases, data_storage=".")


def test_reporttemplate_no_items():
    template = ReportTemplate(title="Empty", items=[])
    assert template.title == "Empty"
    assert template.items == []
    reqs = template.get_requirements()
    expected_keys = ["volume_mesh", "surface_mesh", "geometry"]
    expected_reqs = {RequirementItem.from_data_key(data_key=k) for k in expected_keys}
    assert set(reqs) == expected_reqs


@pytest.mark.usefixtures("mock_detect_latex_compiler")
def test_filepaths_format():
    report_doc = ReportDoc(title="test_doc")
    tex = report_doc.doc.dumps()
    lines = tex.split("\n")
    for line in lines:
        line = line.lstrip()
        if line.startswith(r"\includegraphics"):
            path_match = re.search(r"\{(.*?)\}", line)
            if path_match:
                path = path_match.group(1)
                assert "\\" not in path
        if line.startswith("Path"):
            assert "\\" not in line
