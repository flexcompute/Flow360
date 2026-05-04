"""
Shared thin resource web API wrapper.
"""

from __future__ import annotations

import json

from flow360.cloud.rest_api import RestApi


class ResourceWebApi:
    """Thin wrapper around a single Flow360 resource endpoint."""

    def __init__(self, interface, resource_id: str):
        self.resource_id = resource_id
        self._api = RestApi(interface.endpoint, id=resource_id)

    @staticmethod
    def _unwrap_data(response):
        """Return response data when REST responses use a top-level data envelope."""
        if isinstance(response, dict) and "data" in response:
            return response["data"]
        return response

    def get_info(self):
        """Fetch resource metadata."""
        return self._unwrap_data(self._api.get())

    def get_simulation_json(self):
        """Fetch the resource simulation JSON payload."""
        response = self._unwrap_data(
            self._api.get(method="simulation/file", params={"type": "simulation"})
        )
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
