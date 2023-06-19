import json
import os

import pytest

from flow360.cloud import http_util
from flow360.cloud.http_util import http
from flow360.environment import Env

here = os.path.dirname(os.path.abspath(__file__))


class MockResponse:
    """generic response to a requests function."""

    status_code = 200

    @staticmethod
    def json():
        return {}

    def raise_for_status(self):
        pass


class MockResponseVersions(MockResponse):
    """response if get_supported_server_versions()"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/versions.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseVolumeMeshesPage(MockResponse):
    """response if VolumeMeshList()"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volumemesh_page_webapi_resp.json")) as fh:
            res = json.load(fh)
            res["data"]["data"] = [item for item in res["data"]["data"] if item["deleted"] == False]
        return res


class MockResponseVolumeMeshesPageWithDeleted(MockResponse):
    """response if VolumeMeshList()"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volumemesh_page_webapi_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseVolumeMeshes(MockResponse):
    """response if VolumeMeshList(limit=None)"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volumemesh_webapi_resp.json")) as fh:
            res = json.load(fh)
            res["data"] = [item for item in res["data"] if item["deleted"] == False]
        return res


class MockResponseVolumeMeshesWithDeleted(MockResponse):
    """response if VolumeMeshList(include=true)"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volumemesh_webapi_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseCase(MockResponse):
    """response if Case(id="00000000-0000-0000-0000-000000000000")"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/case_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseVolumeMesh(MockResponse):
    """response if VolumeMesh(id="00000000-0000-0000-0000-000000000000")"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volume_mesh_meta.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseCaseRuntimeParams(MockResponse):
    """response if Case(id="00000000-0000-0000-0000-000000000000").params"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/case_params_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseCaseSubmit(MockResponse):
    """response if Case.submit() volume_mesh_id="00000000-0000-0000-0000-000000000000" """

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/case_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseInfoNotFound(MockResponse):
    """response if web.getinfo(case_id) and case_id not found"""

    status_code = 404

    @staticmethod
    def json():
        return {"data": None}


RESPONSE_MAP = {
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff/case": MockResponseCaseSubmit,
    "/volumemeshes/00000000-0000-0000-0000-000000000000/case": MockResponseCaseSubmit,
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff": MockResponseVolumeMesh,
    "/volumemeshes/00000000-0000-0000-0000-000000000000": MockResponseVolumeMesh,
    "/cases/00000000-0000-0000-0000-000000000000/runtimeParams": MockResponseCaseRuntimeParams,
    "/cases/00000000-0000-0000-0000-000000000000": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb/runtimeParams": MockResponseCaseRuntimeParams,
    "/cases/c58e7a75-e349-476a-9020-247af6b2e92b": MockResponseCase,
    "-python-client-v2": MockResponseVersions,
}


def mock_webapi(url, params):
    method = url.split("flow360")[-1]
    print("<><><><><><<><<><<><><>><<>")

    print(method)

    if method in RESPONSE_MAP.keys():
        return RESPONSE_MAP[method]()

    if method.startswith("-python-client-v2"):
        return MockResponseVersions

    if method.startswith("/volumemeshes/page"):
        if params["includeDeleted"]:
            return MockResponseVolumeMeshesPageWithDeleted()
        return MockResponseVolumeMeshesPage()

    if method.startswith("/volumemeshes"):
        if params["includeDeleted"]:
            return MockResponseVolumeMeshesWithDeleted()
        return MockResponseVolumeMeshes()

    else:
        return MockResponseInfoNotFound()


# monkeypatched requests.get moved to a fixture
@pytest.fixture
def mock_response(monkeypatch):
    """Requests.get() mocked to return {'mock_key':'mock_response'}."""

    def get_response(url: str, params=None, **kwargs) -> str:
        """Get the method path from a full url."""
        preamble = Env.current.web_api_endpoint
        method = url.split(preamble)[-1]

        print(f"calling this mock, {url} {kwargs}")

        res = mock_webapi(method, params)
        print(f"status code: {res.status_code}")
        return mock_webapi(method, params)

    class MockRequests:
        def get(self, url, **kwargs):
            return get_response(url, **kwargs)

        def post(self, url, **kwargs):
            return get_response(url, **kwargs)

    monkeypatch.setattr(
        http_util, "api_key_auth", lambda: {"Authorization": None, "Application": "FLOW360"}
    )
    monkeypatch.setattr(http, "session", MockRequests())
