import json

import pytest

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.component.project_utils import ProjectRecords


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_showing_remote_filtered_projects(mock_id, mock_response):
    _api = RestApi(ProjectInterface.endpoint, id=None)
    resp = _api.get()
    all_projects = ProjectRecords.model_validate({"records": resp["records"]})
    with open("ref/ref_all_projects.json", "r") as f:
        ref_all_projects = ProjectRecords.model_validate(json.load(f))
    assert all_projects == ref_all_projects
