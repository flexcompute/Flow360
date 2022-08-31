"""
Surface mesh component
"""
from datetime import datetime

from pydantic import BaseModel, Extra, Field

from flow360.cloud.http_util import http


class SurfaceMesh(BaseModel, extra=Extra.allow):
    """
    Surface mesh component
    """

    surface_mesh_id: str = Field(..., alias="id")
    name: str
    status: str
    user_id: str = Field(alias="userId")
    solver_version: str = Field(alias="solverVersion")
    config: str
    created_at: str = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    updated_by: str = Field(alias="updatedBy")

    @classmethod
    def from_cloud(cls, mesh_id: str):
        """
        Get surface mesh info from cloud
        :param mesh_id:
        :return:
        """
        resp = http.get(f"surfacemeshes/{mesh_id}")
        if resp.status_code == 200:
            return cls(**resp.json()["data"])
        return None
