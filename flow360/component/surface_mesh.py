"""
Surface mesh component
"""
import json
import os
from typing import Optional

from pydantic import Extra, Field

from flow360.cloud.http_util import http
from flow360.cloud.s3_utils import S3TransferType
from flow360.component.flow360_base_model import Flow360BaseModel


class SurfaceMesh(Flow360BaseModel, extra=Extra.allow):
    """
    Surface mesh component
    """

    surface_mesh_id: Optional[str] = Field(alias="id")
    status: Optional[str]
    config: Optional[str]

    def download(self, file_name: str, to_file=".", keep_folder: bool = True):
        """
        Download file from surface mesh
        :param file_name:
        :param to_file:
        :param keep_folder:
        :return:
        """
        assert self.surface_mesh_id
        S3TransferType.SURFACE_MESH.download_file(
            self.surface_mesh_id, file_name, to_file, keep_folder
        )

    def download_log(self, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download("logs/flow360_surface_mesh.user.log", to_file, keep_folder)

    def submit(self):
        """
        Submit surface mesh
        :return:
        """
        assert self.surface_mesh_id
        http.post(
            f"surfacemeshes/{self.surface_mesh_id}/completeUpload?fileName={self.user_upload_file_name}"
        )

    @classmethod
    def from_cloud(cls, surface_mesh_id: str):
        """
        Get surface mesh info from cloud
        :param surface_mesh_id:
        :return:
        """
        resp = http.get(f"surfacemeshes/{surface_mesh_id}")
        if resp:
            return cls(**resp)
        return None

    @classmethod
    def from_file(
        cls, surface_mesh_name: str, file_name: str, solver_version: str = None, tags: [str] = None
    ):
        """
        Create a surface mesh from a local file
        :param surface_mesh_name:
        :param file_name:
        :param solver_version:
        :param tags:
        :return:
        """
        data = {
            "name": surface_mesh_name
            if surface_mesh_name
            else os.path.splitext(os.path.basename(file_name))[0],
            "tags": tags,
        }
        if solver_version:
            data["solverVersion"] = solver_version
        resp = http.post("surfacemeshes", data)
        if resp:
            mesh = cls(**resp)
            _, ext = os.path.splitext(file_name)
            mesh.user_upload_file_name = f"surfaceMesh{ext}"
            S3TransferType.SURFACE_MESH.upload_file(
                mesh.surface_mesh_id, mesh.user_upload_file_name, file_name
            )
            return mesh
        return None

    # pylint: disable=too-many-arguments
    @classmethod
    def from_geometry(
        cls,
        surface_mesh_name: str,
        geometry_file: str,
        converter_json_file: str,
        solver_version: str = None,
        tags: [str] = None,
    ):
        """
        Create surface mesh from geometry file
        :param surface_mesh_name:
        :param geometry_file:
        :param converter_json_file:
        :param solver_version:
        :param tags:
        :return:
        """
        _, ext = os.path.splitext(geometry_file)
        assert ext == ".csm"
        with open(converter_json_file, "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)
        data = {
            "name": surface_mesh_name
            if surface_mesh_name
            else os.path.splitext(os.path.basename(geometry_file))[0],
            "tags": tags,
            "config": json.dumps(json_content),
        }
        if solver_version:
            data["solverVersion"] = solver_version
        resp = http.post("surfacemeshes", data)
        if resp:
            mesh = cls(**resp)
            S3TransferType.SURFACE_MESH.upload_file(
                mesh.surface_mesh_id, "geometry.csm", geometry_file
            )
            mesh.user_upload_file_name = "geometry.csm"
            return mesh
        return None
