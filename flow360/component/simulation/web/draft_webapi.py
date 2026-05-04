"""
Thin draft web API wrapper.
"""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface
from flow360.component.simulation.web.resource_webapi import ResourceWebApi


class DraftWebApi(ResourceWebApi):
    """Thin wrapper around draft endpoints."""

    def __init__(self, draft_id: str):
        self.draft_id = draft_id
        super().__init__(DraftInterface, draft_id)

    @classmethod
    def list_records(cls, project_id: str):
        """List draft records for a project."""
        api = RestApi(DraftInterface.endpoint)
        response = api.get(params={"projectId": project_id})
        return response.get("records", [])
