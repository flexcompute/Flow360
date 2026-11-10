import pytest

from flow360.cloud.http_util import http
from flow360.component.case import Case
from flow360.exceptions import Flow360RuntimeError
from tests.mock_server import MockResponse


class MockResponseCaseInfo(MockResponse):
    """v2 case info carrying the given statuses."""

    def __init__(self, display_status, status):
        self._body = {"data": {"displayStatus": display_status, "status": status}}

    def json(self):
        return self._body


def mock_case_endpoints(monkeypatch, display_status, status):
    """Serve the case info and record every PUT request."""
    put_requests = []

    def record_put(url, json, **_):
        put_requests.append((url, json))
        return MockResponse()

    monkeypatch.setattr(
        http.session, "get", lambda **_: MockResponseCaseInfo(display_status, status)
    )
    monkeypatch.setattr(http.session, "put", record_put)
    return put_requests


def test_case_stop_sends_empty_body_to_v2_endpoint(mock_id, monkeypatch):
    put_requests = mock_case_endpoints(monkeypatch, "running", "preprocessing")

    Case(mock_id).stop()

    assert len(put_requests) == 1
    url, body = put_requests[0]
    assert url.endswith(f"/v2/cases/{mock_id}/stop")
    assert body == {}


def test_case_stop_rejects_case_that_has_not_started(mock_id, monkeypatch):
    put_requests = mock_case_endpoints(monkeypatch, "submitted", "queued")

    with pytest.raises(
        Flow360RuntimeError, match=r"has not started yet \(status: queued\).*delete it"
    ):
        Case(mock_id).stop()

    assert put_requests == []


def test_case_stop_rejects_finished_case(mock_id, monkeypatch):
    put_requests = mock_case_endpoints(monkeypatch, "completed", "completed")

    with pytest.raises(
        Flow360RuntimeError, match="is completed; only a running case can be stopped"
    ):
        Case(mock_id).stop()

    assert put_requests == []
