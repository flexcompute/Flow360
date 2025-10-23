import json
import os

import pytest

from flow360.cloud import http_util
from flow360.cloud.http_util import http
from flow360.cloud.s3_utils import CloudFileNotFoundError
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


class MockResponseGeometrySimConfigV2(MockResponse):
    """response for Geometry(id="00000000-0000-0000-0000-000000000000")'s simulation json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/geometry_birth_simulation_json.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseSimulationJsonFile(MockResponse):
    """response for Case(id="00000000-0000-0000-0000-000000000000").params simulation json"""

    @staticmethod
    def json():
        raise CloudFileNotFoundError(
            error_response={"Error": {"Message": "file not found: simulation.json"}},
            operation_name="download",
        )


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


class MockResponseProject(MockResponse):
    """response for Project.from_cloud(id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectFromVM(MockResponse):
    """response for Project.from_cloud(id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_from_vm_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseAllProjects(MockResponse):
    """response of projects get"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/get_projects_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectTree(MockResponse):
    """response for Project.from_cloud(id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")'s tree json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_get_tree_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectFromVMTree(MockResponse):
    """response for Project.from_cloud(id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")'s tree json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_from_vm_get_tree_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectGeometry(MockResponse):
    """response for Geometry.from_cloud(id="geo-2877e124-96ff-473d-864b-11eec8648d42")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_geometry_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectGeometrySimConfig(MockResponse):
    """response for Geometry.from_cloud(id="geo-2877e124-96ff-473d-864b-11eec8648d42")'s simualtion json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_geometry_simulation_json_resp.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectSurfaceMesh(MockResponse):
    """response for SurfaceMesh.from_cloud(id="sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_surface_mesh_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectSurfaceMeshSimConfig(MockResponse):
    """response for SurfaceMesh.from_cloud(id="sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a")'s simualtion json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_surface_mesh_simulation_json_resp.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectVolumeMesh(MockResponse):
    """response for VolumeMesh.from_cloud(id="vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_volume_mesh_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectVolumeMeshSimConfig(MockResponse):
    """response for VolumeMesh.from_cloud(id="vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3")'s simualtion json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_volume_mesh_simulation_json_resp.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectFromVMVolumeMeshMeta(MockResponse):
    """response for VolumeMesh.from_cloud(id="vm-bff35714-41b1-4251-ac74-46a40b95a330")'s meta json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_from_vm_volume_mesh_meta_resp.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectFromVMVolumeMeshSimConfig(MockResponse):
    """response for VolumeMesh.from_cloud(id="vm-bff35714-41b1-4251-ac74-46a40b95a330")'s simualtion json"""

    @staticmethod
    def json():
        with open(
            os.path.join(
                here, "data/mock_webapi/project_from_vm_volume_mesh_simulation_json_resp.json"
            )
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectCase(MockResponse):
    """response for Case.from_cloud(id="case-69b8c249-fce5-412a-9927-6a79049deebb")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_case_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectFromVMCase(MockResponse):
    """response for Case.from_cloud(id="case-f7480884-4493-4453-9a27-dd5f8498c608")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_from_vm_case_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectCaseFork(MockResponse):
    """response for Case.from_cloud(id="case-69b8c249-fce5-412a-9927-6a79049deebb")'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_case_fork_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectCaseSimConfig(MockResponse):
    """response for Case.from_cloud(id="case-69b8c249-fce5-412a-9927-6a79049deebb")'s simulation json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/case-69b8c249-fce5-412a-9927-6a79049deebb/simulation.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectCaseForkSimConfig(MockResponse):
    """response for Case.from_cloud(id="case-84d4604e-f3cd-4c6b-8517-92a80a3346d3")'s simulation json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_case_fork_simulation_json_resp.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectRunCase(MockResponse):
    """response for project.run_case(params = params)'s meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/project_case_fork_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectPath(MockResponse):
    """response for Project(id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")'s path"""

    def __init__(self, *args, params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._params = params

    def json(self):
        with open(os.path.join(here, "data/mock_webapi/project_path_to_case_fork_resp.json")) as fh:
            # The json file contains all the assets along the path from geometry to the forked case.
            # Thus, to get the path to the intermediate asset, the extra items are removed from the response.
            res = json.load(fh)
            if self._params["itemType"] == "VolumeMesh":
                res["data"]["cases"] = []
            if self._params["itemType"] == "SurfaceMesh":
                res["data"]["volumeMesh"] = None
                res["data"]["cases"] = []
            if self._params["itemType"] == "case-69b8c249-fce5-412a-9927-6a79049deebb":
                res["data"]["cases"] = []
        return res


class MockResponseDraftSubmit(MockResponse):
    """response for Project(id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")'s path to Fork Case json"""

    def __init__(self, *args, params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._params = params

    def json(self):
        res = None
        if self._params["name"] == "VolumeMesh":
            with open(
                os.path.join(here, "data/mock_webapi/project_draft_volume_mesh_submit_resp.json")
            ) as fh:
                res = json.load(fh)

        if self._params["name"] == "Case":
            with open(
                os.path.join(here, "data/mock_webapi/project_draft_case_fork_submit_resp.json")
            ) as fh:
                res = json.load(fh)
        return res


class MockResponseDraftVolumeMeshRun(MockResponse):
    """response for Project(id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")'s path to Fork Case json"""

    @staticmethod
    def json():
        with open(
            os.path.join(here, "data/mock_webapi/project_draft_run_to_volume_mesh.json")
        ) as fh:
            res = json.load(fh)
        return res


class MockResponseProjectPatchDraftSubmit(MockResponse):

    def __init__(self, *args, params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._params = params

    def json(self):
        with open(os.path.join(here, "data/mock_webapi/project_meta_resp.json")) as fh:
            res = json.load(fh)
        res["data"]["lastOpenItemId"] = self._params["lastOpenItemId"]
        res["data"]["lastOpenItemType"] = self._params["lastOpenItemType"]
        return res


class MockResponseReportSubmit(MockResponse):
    """response for report_template.create_in_cloud's meta json"""

    @staticmethod
    def json():
        with open(os.path.join(here, "data/mock_webapi/report_meta_resp.json")) as fh:
            res = json.load(fh)
        return res


GET_RESPONSE_MAP = {
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff": MockResponseVolumeMesh,
    "/volumemeshes/00000000-0000-0000-0000-000000000000": MockResponseVolumeMesh,
    "/v2/geometries/00000000-0000-0000-0000-000000000000": MockResponseGeometryV2,
    "/v2/projects/prj-29e35434-2148-47c8-b548-58b479c37b99": MockResponseGeometryProjectV2,
    "/v2/geometries/00000000-0000-0000-0000-000000000000/simulation/file": MockResponseGeometrySimConfigV2,
    "/cases/00000000-0000-0000-0000-000000000000/runtimeParams": MockResponseCaseRuntimeParams,
    "/v2/cases/00000000-0000-0000-0000-000000000000/file?filename=simulation.json": MockResponseSimulationJsonFile,
    "/cases/00000000-0000-0000-0000-000000000000": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb": MockResponseCase,
    "/cases/00112233-4455-6677-8899-bbbbbbbbbbbb/runtimeParams": MockResponseCaseRuntimeParams,
    "/cases/c58e7a75-e349-476a-9020-247af6b2e92b": MockResponseCase,
    "-python-client-v2": MockResponseVersions,
    "/account": MockResponseOrganizationAccounts,
    "/folders/items/folder-3834758b-3d39-4a4a-ad85-710b7652267c/metadata": MockResponseFolderRootMetadata,
    "/folders/items/folder-4da3cdd0-c5b6-4130-9ca1-196237322ab9/metadata": MockResponseFolderNestedMetadata,
    "/v2/projects/prj-41d2333b-85fd-4bed-ae13-15dcb6da519e": MockResponseProject,
    "/v2/projects/prj-99cc6f96-15d3-4170-973c-a0cced6bf36b": MockResponseProjectFromVM,
    "/v2/projects/prj-41d2333b-85fd-4bed-ae13-15dcb6da519e/tree": MockResponseProjectTree,
    "/v2/projects/prj-99cc6f96-15d3-4170-973c-a0cced6bf36b/tree": MockResponseProjectFromVMTree,
    "/v2/geometries/geo-2877e124-96ff-473d-864b-11eec8648d42": MockResponseProjectGeometry,
    "/v2/geometries/geo-2877e124-96ff-473d-864b-11eec8648d42/simulation/file": MockResponseProjectGeometrySimConfig,
    "/v2/surface-meshes/sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a": MockResponseProjectSurfaceMesh,
    "/v2/surface-meshes/sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a/simulation/file": MockResponseProjectSurfaceMeshSimConfig,
    "/v2/volume-meshes/vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3": MockResponseProjectVolumeMesh,
    "/v2/volume-meshes/vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3/simulation/file": MockResponseProjectVolumeMeshSimConfig,
    "/v2/volume-meshes/vm-bff35714-41b1-4251-ac74-46a40b95a330": MockResponseProjectFromVMVolumeMeshMeta,
    "/v2/volume-meshes/vm-bff35714-41b1-4251-ac74-46a40b95a330/simulation/file": MockResponseProjectFromVMVolumeMeshSimConfig,
    "/cases/case-69b8c249-fce5-412a-9927-6a79049deebb": MockResponseProjectCase,
    "/v2/cases/case-69b8c249-fce5-412a-9927-6a79049deebb/simulation/file": MockResponseProjectCaseSimConfig,
    "/cases/case-f7480884-4493-4453-9a27-dd5f8498c608": MockResponseProjectFromVMCase,
    "/cases/case-84d4604e-f3cd-4c6b-8517-92a80a3346d3": MockResponseProjectCaseFork,
    "/v2/cases/case-84d4604e-f3cd-4c6b-8517-92a80a3346d3/simulation/file": MockResponseProjectCaseForkSimConfig,
    "/v2/projects": MockResponseAllProjects,
}

PUT_RESPONSE_MAP = {
    "/folders/move": MockResponseFolderSubmit,
}

POST_RESPONSE_MAP = {
    "/volumemeshes/00112233-4455-6677-8899-aabbccddeeff/case": MockResponseCaseSubmit,
    "/volumemeshes/00000000-0000-0000-0000-000000000000/case": MockResponseCaseSubmit,
    "/folders": MockResponseFolderSubmit,
    "/v2/drafts/vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3/simulation/file": MockResponseProjectVolumeMeshSimConfig,
    "/v2/drafts/vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3/run": MockResponseProjectVolumeMesh,
    "/v2/drafts/case-84d4604e-f3cd-4c6b-8517-92a80a3346d3/simulation/file": MockResponseProjectCaseForkSimConfig,
    "/v2/drafts/case-84d4604e-f3cd-4c6b-8517-92a80a3346d3/run": MockResponseProjectCaseFork,
    "/v2/report": MockResponseReportSubmit,
}


def mock_webapi(type, url, params):
    method = url.split("flow360")[-1]
    print("<><><><><><><><><><><><><><><>")

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

        if method.endswith("/path"):
            return MockResponseProjectPath(params=params)

    elif type == "put":
        if method == "/folders/move":
            return MockResponseFolderMove(params=params)

        if method in PUT_RESPONSE_MAP.keys():
            return PUT_RESPONSE_MAP[method]()

    elif type == "post":
        if method == "/folders":
            return MockResponseFolderSubmit(params=params)

        if method == "/v2/drafts":
            return MockResponseDraftSubmit(params=params)

        if method in POST_RESPONSE_MAP.keys():
            return POST_RESPONSE_MAP[method]()

    elif type == "patch":
        if method.startswith("/v2/projects"):
            return MockResponseProjectPatchDraftSubmit(params=params)

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

        def patch(self, url, json=None, **kwargs):
            return get_response(url, type="patch", params=json, **kwargs)

        def put(self, url, json=None, **kwargs):
            return get_response(url, type="put", params=json, **kwargs)

        def post(self, url, json=None, **kwargs):
            return get_response(url, type="post", params=json, **kwargs)

    monkeypatch.setattr(
        http_util, "api_key_auth", lambda: {"Authorization": None, "Application": "FLOW360"}
    )
    monkeypatch.setattr(http, "session", MockRequests())
