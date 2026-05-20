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


class ResourceWebApi:
    """Thin wrapper around a single Flow360 resource endpoint."""

    def __init__(self, interface, resource_id: str):
        self.resource_id = resource_id
        self._api = RestApi(interface.endpoint, id=resource_id)

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
        api = RestApi(DraftInterface.endpoint)
        response = api.get(params={"projectId": project_id})
        return response.get("records", [])
