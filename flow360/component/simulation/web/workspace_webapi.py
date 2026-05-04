"""Thin workspace web API wrapper."""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import WorkspaceInterface


class WorkspaceWebApi:
    """Thin wrapper around workspace endpoints."""

    @classmethod
    def list_records(cls):
        """List available workspace records."""
        api = RestApi(WorkspaceInterface.endpoint)
        response = api.get()
        if isinstance(response, list):
            return response
        return response.get("data", [])

    @classmethod
    def get_workspace_id_for_root_folder(cls, root_folder_id: str) -> str | None:
        """Return the workspace ID that owns a root folder, if available."""
        for record in cls.list_records():
            if record.get("rootFolderId") == root_folder_id:
                return record.get("id")
        return None
