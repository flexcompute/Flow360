from flow360.cloud.http_util import http
from flow360.component.case import Case
from tests.mock_server import MockResponse


def test_case_delete_uses_v2_endpoint(mock_id, monkeypatch):
    deleted_urls = []

    def record_delete(url, **_):
        deleted_urls.append(url)
        return MockResponse()

    monkeypatch.setattr(http.session, "delete", record_delete)

    Case(mock_id).delete()

    assert len(deleted_urls) == 1
    assert deleted_urls[0].endswith(f"/v2/cases/{mock_id}")
