"""
Thin project web API wrapper.
"""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.environment import current_environment


class ProjectWebApi:
    """Thin wrapper around project endpoints."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._api = RestApi(
            ProjectInterface.endpoint, id=project_id, environment_provider=current_environment
        )

    def get_info(self):
        """Get project metadata."""

        return self._api.get()

    def get_tree(self):
        """Get project asset tree records."""

        return self._api.get(method="tree")

    def get_path(self, item_id: str, item_type: str):
        """Get the path from the project root to an item."""

        return self._api.get(method="path", params={"itemId": item_id, "itemType": item_type})

    def rename(self, name: str):
        """Rename project."""

        return self._api.patch({"name": name})

    def delete(self):
        """Delete project."""

        return self._api.delete()
