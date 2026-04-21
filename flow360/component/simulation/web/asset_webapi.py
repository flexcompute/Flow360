"""
Thin asset web API wrappers.
"""

from __future__ import annotations

import json

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import (
    CaseInterface,
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

    def post(self, json, path=None, method=None):
        return self._api.post(json=json, path=path, method=method)

    def put(self, json, path=None, method=None):
        return self._api.put(json=json, path=path, method=method)

    def patch(self, json, path=None, method=None):
        return self._api.patch(json=json, path=path, method=method)

    def delete(self, path=None, method=None):
        return self._api.delete(path=path, method=method)


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
        self._files_api = RestApi(CaseInterface.endpoint, id=asset_id)

    def list_files(self):
        return self._unwrap_data(self._files_api.get(method="files"))

    def download_file(self, remote_file_name, *, to_file=None, to_folder=".", overwrite=False):
        return CaseInterface.s3_transfer_method.download_file(
            self.asset_id,
            remote_file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
            verbose=False,
        )
