"""
Thin project web API wrapper.
"""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface


class ProjectWebApi:
    """Thin wrapper around project endpoints."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._api = RestApi(ProjectInterface.endpoint, id=project_id)

    def get_info(self):
        return self._api.get()

    def get_tree(self):
        return self._api.get(method="tree")

    def get_path(self, item_id: str, item_type: str):
        return self._api.get(method="path", params={"itemId": item_id, "itemType": item_type})

    def get_dependency(self):
        return self._api.get(method="dependency")

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

