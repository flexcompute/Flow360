import json

import pytest

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.component.simulation.web.project_records import ProjectRecords
from flow360.environment import current_environment


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_showing_remote_filtered_projects(mock_id, mock_response):
    _api = RestApi(ProjectInterface.endpoint, id=None, environment_provider=current_environment)
    resp = _api.get()
    all_projects = ProjectRecords.model_validate({"records": resp["records"]})
    with open("ref/ref_all_projects.json", "r") as f:
        ref_all_projects = ProjectRecords.model_validate(json.load(f))
    assert all_projects == ref_all_projects


def test_project_records_tolerate_null_statistics():
    # On-premises deployments return null statistics for projects without assets.
    record = {
        "name": "empty project",
        "id": "prj-000000000000",
        "tags": [],
        "statistics": None,
        "createdAt": "2026-08-07T00:00:00.000Z",
        "rootItemType": "Geometry",
    }
    records = ProjectRecords.model_validate({"records": [record]})
    assert records.records[0].statistics is None
    assert "empty project" in str(records)
