"""
Thin asset web API wrappers.
"""

from __future__ import annotations

import json

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import (
    CaseInterfaceV2,
    GeometryInterface,
    SurfaceMeshInterfaceV2,
    VolumeMeshInterfaceV2,
)


class AssetWebApi:
    """Thin wrapper around a single asset endpoint."""

    def __init__(self, interface, asset_id: str):
        self.asset_id = asset_id
        self._api = RestApi(interface.endpoint, id=asset_id)

    @staticmethod
    def _unwrap_data(response):
        if isinstance(response, dict) and "data" in response:
            return response["data"]
        return response

    def get_info(self):
        return self._unwrap_data(self._api.get())

    def get_simulation_json(self):
        response = self._unwrap_data(
            self._api.get(method="simulation/file", params={"type": "simulation"})
        )
        if isinstance(response, dict) and "simulationJson" in response:
            response = response["simulationJson"]
        if isinstance(response, str):
            return json.loads(response)
        return response

    def get(self, path=None, method=None, json=None, params=None):
        return self._api.get(path=path, method=method, json=json, params=params)


class GeometryWebApi(AssetWebApi):
    def __init__(self, asset_id: str):
        super().__init__(GeometryInterface, asset_id)


class SurfaceMeshWebApi(AssetWebApi):
    def __init__(self, asset_id: str):
        super().__init__(SurfaceMeshInterfaceV2, asset_id)


class VolumeMeshWebApi(AssetWebApi):
    def __init__(self, asset_id: str):
        super().__init__(VolumeMeshInterfaceV2, asset_id)


class CaseWebApi(AssetWebApi):
    def __init__(self, asset_id: str):
        super().__init__(CaseInterfaceV2, asset_id)
