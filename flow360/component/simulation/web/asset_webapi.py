"""
Thin V2 resource web API wrappers.
"""

from __future__ import annotations

import json

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import (
    CaseInterfaceV2,
    DraftInterface,
    GeometryInterface,
    SurfaceMeshInterfaceV2,
    VolumeMeshInterfaceV2,
)
from flow360.environment import current_environment


class ResourceWebApi:
    """Thin wrapper around a single Flow360 resource endpoint."""

    def __init__(self, interface, resource_id: str):
        self.resource_id = resource_id
        self._interface = interface
        self._api = RestApi(
            interface.endpoint, id=resource_id, environment_provider=current_environment
        )

    def get_info(self):
        """Fetch resource metadata."""
        return self._api.get()

    def get_simulation_params(self):
        """Fetch the resource SimulationParams payload."""
        response = self._api.get(method="simulation/file", params={"type": "simulation"})
        if isinstance(response, dict) and "simulationJson" in response:
            response = response["simulationJson"]
        if isinstance(response, str):
            return json.loads(response)
        return response

    def get(
        self, path=None, method=None, json=None, params=None
    ):  # pylint: disable=redefined-outer-name
        """Delegate specialized GET calls to the underlying REST API."""
        return self._api.get(path=path, method=method, json=json, params=params)

    def post(self, json, path=None, method=None):  # pylint: disable=redefined-outer-name
        """Delegate specialized POST calls to the underlying REST API."""
        return self._api.post(json=json, path=path, method=method)

    def patch(self, json, path=None, method=None):  # pylint: disable=redefined-outer-name
        """Delegate specialized PATCH calls to the underlying REST API."""
        return self._api.patch(json=json, path=path, method=method)

    def delete(self, path=None, method=None):
        """Delegate DELETE calls to the underlying REST API."""
        return self._api.delete(path=path, method=method)

    def rename(self, name: str):
        """Rename resource."""
        return self.patch({"name": name})

    def list_files(self):
        """List files available for this resource."""
        return self._api.get(method="files") or []

    def get_download_file_list(self):
        """Return the files available for SDK download helpers."""
        return self.list_files()

    def download_file(self, file_name, *, to_file=None, to_folder=".", overwrite=True):
        """Download a resource file by remote name."""
        return self._interface.s3_transfer_method.download_file(
            self.resource_id,
            file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
        )


class GeometryWebApi(ResourceWebApi):
    """Thin geometry web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(GeometryInterface, asset_id)


class SurfaceMeshWebApi(ResourceWebApi):
    """Thin surface mesh web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(SurfaceMeshInterfaceV2, asset_id)


class VolumeMeshWebApi(ResourceWebApi):
    """Thin volume mesh web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(VolumeMeshInterfaceV2, asset_id)


class CaseWebApi(ResourceWebApi):
    """Thin case web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(CaseInterfaceV2, asset_id)


class DraftWebApi(ResourceWebApi):
    """Thin draft web API wrapper."""

    def __init__(self, draft_id: str):
        super().__init__(DraftInterface, draft_id)

    @classmethod
    def list_records(cls, project_id: str):
        """List draft records for a project."""
        api = RestApi(DraftInterface.endpoint, environment_provider=current_environment)
        response = api.get(params={"projectId": project_id})
        return response.get("records", [])

    @classmethod
    def create(  # pylint: disable=too-many-arguments
        cls,
        *,
        name,
        project_id,
        source_item_id,
        source_item_type,
        solver_version,
        fork_case,
    ):
        """Create a draft from an existing project item."""
        api = RestApi(DraftInterface.endpoint, environment_provider=current_environment)
        return api.post(
            {
                "name": name,
                "projectId": project_id,
                "sourceItemId": source_item_id,
                "sourceItemType": source_item_type,
                "solverVersion": solver_version,
                "forkCase": fork_case,
            }
        )

    def set_simulation_params(self, simulation_params):
        """Replace the draft SimulationParams payload."""
        return self.post(
            {
                "data": json.dumps(simulation_params),
                "type": "simulation",
                "version": "",
            },
            method="simulation/file",
        )

    def run(self, *, up_to, use_in_house=False, use_gai=False, start_from=None):
        """Run the draft up to the requested asset type."""
        payload = {
            "upTo": up_to,
            "useInHouse": use_in_house,
            "useGai": use_gai,
        }
        if start_from is not None:
            payload["forceCreationConfig"] = {"startFrom": start_from}
        return self.post(payload, method="run")


def get_resource_webapi_class(resource_type: str):
    """Return the lightweight V2 web API class for a resource type."""
    webapi_by_type = {
        "Draft": DraftWebApi,
        "Geometry": GeometryWebApi,
        "SurfaceMesh": SurfaceMeshWebApi,
        "VolumeMesh": VolumeMeshWebApi,
        "Case": CaseWebApi,
    }
    try:
        return webapi_by_type[resource_type]
    except KeyError as error:
        raise ValueError(f"Web API is not supported for {resource_type} resources.") from error
