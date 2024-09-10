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
            res["data"]["data"] = [item for item in res["data"]["data"] if not item["deleted"]]
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
            res["data"] = [item for item in res["data"] if not item["deleted"]]
        return res


class MockResponseVolumeMeshesWithDeleted(MockResponse):
    """response if VolumeMeshList(include=true)"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/volumemesh_webapi_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseOrganizationAccounts(MockResponse):
    """response to retrieving shared account list"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/organization_accounts_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseClientAccounts(MockResponse):
    """response to retrieving shared account list"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/client_accounts_resp.json")) as fh:
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


class MockResponseGeometryV2(MockResponse):
    """response if Geometry(id="00000000-0000-0000-0000-000000000000")"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/geometry_v2_meta.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseGeometryProjectV2(MockResponse):
    """response for Geometry(id="00000000-0000-0000-0000-000000000000")'s project"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/geometry_project_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseGeometryProjectSimConfigV2(MockResponse):
    """response for Geometry(id="00000000-0000-0000-0000-000000000000")'s simulation json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/geometry_draft_simulation_json.json")) as fh:
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


class MockResponseFolderSubmit(MockResponse):
    """response if Folder.submit()"""

    def __init__(self, *args, params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._params = params

    def json(self):
        print(self._params)
        resp_file = None
        if self._params["parentFolderId"] == "ROOT.FLOW360":
            resp_file = "data/mock_webapi/folder_root_submit_resp.json"
        elif self._params["parentFolderId"] == "folder-3834758b-3d39-4a4a-ad85-710b7652267c":
            resp_file = "data/mock_webapi/folder_nested_submit_resp.json"
        with open(os.path.join(here, resp_file)) as fh:
            res = json.load(fh)
        return res


class MockResponseFolderRootMetadata(MockResponse):
    """response if Folder.info for folder at root level"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/folder_at_root_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseFolderNestedMetadata(MockResponse):
    """response if Folder.info for folder at nested level"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/folder_nested_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseFolderMove(MockResponse):
    """response if moving to folder"""

    def __init__(self, *args, params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._params = params

    def json(self):
        print(self._params)
        if (
            self._params["items"][0]["id"] == "folder-3834758b-3d39-4a4a-ad85-710b7652267c"
            and self._params["items"][0]["type"] == "folder"
        ):
            return super().json()
        elif (
            self._params["items"][0]["id"] == "00000000-0000-0000-0000-000000000000"
            and self._params["items"][0]["type"] == "case"
        ):
            return super().json()

        raise ValueError


class MockResponseInfoNotFound(MockResponse):
    """response if web.getinfo(case_id) and case_id not found"""

    status_code = 404

    @staticmethod
    def json():
        return {"data": None}


GET_RESPONSE_MAP = {
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff": MockResponseVolumeMesh,
    "/volumemeshes/00000000-0000-0000-0000-000000000000": MockResponseVolumeMesh,
    "/v2/geometries/00000000-0000-0000-0000-000000000000": MockResponseGeometryV2,
    "/v2/projects/prj-29e35434-2148-47c8-b548-58b479c37b99": MockResponseGeometryProjectV2,
    "/v2/drafts/draft-e6cf57bd-07bc-43af-b107-a5e689dfd979/simulation-config": MockResponseGeometryProjectSimConfigV2,
    "/cases/00000000-0000-0000-0000-000000000000/runtimeParams": MockResponseCaseRuntimeParams,
    "/cases/00000000-0000-0000-0000-000000000000": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb/runtimeParams": MockResponseCaseRuntimeParams,
    "/cases/c58e7a75-e349-476a-9020-247af6b2e92b": MockResponseCase,
    "-python-client-v2": MockResponseVersions,
    "/account": MockResponseOrganizationAccounts,
    "/folders/items/folder-3834758b-3d39-4a4a-ad85-710b7652267c/metadata": MockResponseFolderRootMetadata,
    "/folders/items/folder-4da3cdd0-c5b6-4130-9ca1-196237322ab9/metadata": MockResponseFolderNestedMetadata,
}

PUT_RESPONSE_MAP = {
    "/folders/move": MockResponseFolderSubmit,
}

POST_RESPONSE_MAP = {
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff/case": MockResponseCaseSubmit,
    "/volumemeshes/00000000-0000-0000-0000-000000000000/case": MockResponseCaseSubmit,
    "/folders": MockResponseFolderSubmit,
}


def mock_webapi(type, url, params):
    method = url.split("flow360")[-1]
    print("<><><><><><<><<><<><><>><<>")

    print(type, method)

    if type == "get":
        if method in GET_RESPONSE_MAP.keys():
            return GET_RESPONSE_MAP[method]()

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

        if method.endswith("/auth/credential"):
            return MockResponseClientAccounts

    elif type == "put":
        if method == "/folders/move":
            return MockResponseFolderMove(params=params)

        if method in PUT_RESPONSE_MAP.keys():
            return PUT_RESPONSE_MAP[method]()

    elif type == "post":
        if method == "/folders":
            return MockResponseFolderSubmit(params=params)

        if method in POST_RESPONSE_MAP.keys():
            return POST_RESPONSE_MAP[method]()

    return MockResponseInfoNotFound()


# monkeypatched requests.get moved to a fixture
@pytest.fixture
def mock_response(monkeypatch):
    """Requests.get() mocked to return {'mock_key':'mock_response'}."""

    def get_response(url: str, type="get", params=None, **kwargs) -> str:
        """Get the method path from a full url."""
        preamble = Env.current.web_api_endpoint
        method = url.split(preamble)[-1]

        print(f"calling this mock, {url} {kwargs}")

        res = mock_webapi(type, method, params)
        print(f"status code: {res.status_code}")
        return res

    class MockRequests:
        def get(self, url, **kwargs):
            return get_response(url, type="get", **kwargs)

        def put(self, url, json=None, **kwargs):
            return get_response(url, type="put", params=json, **kwargs)

        def post(self, url, json=None, **kwargs):
            return get_response(url, type="post", params=json, **kwargs)

    monkeypatch.setattr(
        http_util, "api_key_auth", lambda: {"Authorization": None, "Application": "FLOW360"}
    )
    monkeypatch.setattr(http, "session", MockRequests())
