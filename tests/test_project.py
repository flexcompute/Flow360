import pydantic as pd
import pytest

import flow360 as fl
from flow360 import log
from flow360.exceptions import Flow360ValueError

log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_from_cloud(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")

    assert project.length_unit == 1 * fl.u.m
    assert project_vm.length_unit == 1 * fl.u.m
    assert isinstance(project._root_asset, fl.Geometry)
    assert isinstance(project_vm._root_asset, fl.VolumeMesh)


@pytest.mark.usefixtures("s3_download_override")
def test_run(mock_response, capsys):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    case = fl.Case.from_cloud("case-84d4604e-f3cd-4c6b-8517-92a80a3346d3")

    params = case.params

    error_msg = r"Input should be 'SurfaceMesh', 'VolumeMesh' or 'Case'"
    with pytest.raises(pd.ValidationError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Geometry")

    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(Case\) "
        r"cannot be later than 'up_to' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Case")

    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")
    case = fl.Case.from_cloud("case-f7480884-4493-4453-9a27-dd5f8498c608")
    params = case.params
    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(SurfaceMesh\) "
        r"must be later than 'source_item_type' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project_vm.run_case(params=params, start_from="SurfaceMesh")
