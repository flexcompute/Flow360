"""
Thin folder web API wrapper.
"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import FolderInterfaceV2


class FolderWebApi:
    """Thin wrapper around folder endpoints."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self._api = RestApi(FolderInterfaceV2.endpoint, id=folder_id)

    def get_info(self):
        """Get folder metadata."""

        return self._api.get()
