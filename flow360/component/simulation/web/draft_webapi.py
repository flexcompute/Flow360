"""
Thin draft web API wrapper.
"""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface


class DraftWebApi:
    """Thin wrapper around draft endpoints."""

    def __init__(self, draft_id: str):
        self.draft_id = draft_id
        self._api = RestApi(DraftInterface.endpoint, id=draft_id)

    @staticmethod
    def _unwrap_data(response):
        """Return response data when REST responses use a top-level data envelope."""
        if isinstance(response, dict) and "data" in response:
            return response["data"]
        return response

    def get_info(self):
        """Fetch draft metadata."""
        return self._unwrap_data(self._api.get())

    @classmethod
    def list_records(cls, project_id: str):
        """List draft records for a project."""
        api = RestApi(DraftInterface.endpoint)
        response = api.get(params={"projectId": project_id})
        return response.get("records", [])

    def get_simulation_json(self):
        """Fetch the draft simulation JSON payload."""
        response = self._api.get(method="simulation/file", params={"type": "simulation"})
        if isinstance(response, dict) and "simulationJson" in response:
            return response["simulationJson"]
        return response

    def get(self, path=None, method=None, json=None, params=None):
        """Delegate specialized GET calls to the underlying REST API."""
        return self._api.get(path=path, method=method, json=json, params=params)
