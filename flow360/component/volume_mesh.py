"""
Surface mesh component
"""
import json
import os.path
from enum import Enum
from typing import Optional, Union

from pydantic import Extra, Field

from flow360.cloud.http_util import http
from flow360.cloud.s3_utils import S3TransferType
from flow360.component.flow360_base_model import Flow360BaseModel
from flow360.component.flow360_solver_params import Flow360MeshParams, Flow360Params


class VolumeMeshLog(Enum):
    """
    Volume mesh log
    """

    USER_LOG = "user.log"
    PY_LOG = "validateFlow360Mesh.py.log"


class VolumeMeshDownloadable(Enum):
    """
    Volume mesh downloadable files
    """

    CONFIG_JSON = "config.json"


class UGRIDEndianness(Enum):
    """
    UGRID endianness
    """

    LITTLE = "little"
    BIG = "big"


class VolumeMeshFileFormat(Enum):
    """
    Volume mesh file format
    """

    UGRID = "aflr3"
    CGNS = "cgns"


class VolumeMesh(Flow360BaseModel, extra=Extra.allow):
    """
    Surface mesh component
    """

    volume_mesh_id: Optional[str] = Field(alias="id")
    name: str = Field(alias="meshName")
    status: Optional[str] = Field(alias="meshStatus")
    surface_mesh_id: Optional[str] = Field(alias="surfaceMeshId")
    mesh_params: Optional[str] = Field(alias="meshParams")

    def download(
        self, file_name: Union[str, VolumeMeshDownloadable], to_file=".", keep_folder: bool = True
    ):
        """
        Download file from surface mesh
        :param file_name:
        :param to_file:
        :param keep_folder:
        :return:
        """
        assert self.volume_mesh_id
        if isinstance(file_name, VolumeMeshDownloadable):
            file_name = file_name.value
        S3TransferType.VOLUME_MESH.download_file(
            self.volume_mesh_id, file_name, to_file, keep_folder
        )

    def download_log(self, log: VolumeMeshLog, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param log:
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download(f"logs/{log.value}", to_file, keep_folder)

    def submit(self):
        """
        Submit surface mesh
        :return:
        """
        assert self.volume_mesh_id
        http.post(
            f"volumemeshes/{self.volume_mesh_id}/completeUpload?fileName={self.user_upload_file_name}"
        )

    @classmethod
    def from_cloud(cls, surface_mesh_id: str):
        """
        Get surface mesh info from cloud
        :param surface_mesh_id:
        :return:
        """
        resp = http.get(f"volumemeshes/{surface_mesh_id}")
        if resp:
            return cls(**resp)
        return None

    # pylint: disable=too-many-arguments
    @classmethod
    def from_surface_mesh(
        cls,
        volume_mesh_name: str,
        surface_mesh_id: str,
        config_file: str,
        tags: [str] = None,
        solver_version=None,
    ):
        """
        Create volume mesh from surface mesh
        :param volume_mesh_name:
        :param surface_mesh_id:
        :param config_file:
        :param tags:
        :param solver_version:
        :return:
        """
        assert volume_mesh_name
        assert os.path.exists(config_file)
        assert surface_mesh_id
        with open(config_file, "r", encoding="utf-8") as config_f:
            json_content = json.load(config_f)
        body = {
            "name": volume_mesh_name,
            "tags": tags,
            "surfaceMeshId": surface_mesh_id,
            "config": json.dumps(json_content),
            "format": "cgns",
        }

        if solver_version:
            body["solverVersion"] = solver_version

        resp = http.post("volumemeshes", body)
        if resp:
            return cls(**resp)
        return None

    # pylint: disable=too-many-arguments
    @classmethod
    def from_ugrid_file(
        cls,
        volume_mesh_name: str,
        file_name: str,
        params: Union[Flow360Params, Flow360MeshParams],
        endianness: UGRIDEndianness = UGRIDEndianness.BIG,
        tags: [str] = None,
        solver_version=None,
    ):
        """
        Create volume mesh from ugrid file
        :param volume_mesh_name:
        :param file_name:
        :param params:
        :param endianness:
        :param tags:
        :param solver_version:
        :return:
        """
        assert volume_mesh_name
        assert os.path.exists(file_name)
        assert params
        assert endianness

        body = {
            "meshName": volume_mesh_name,
            "meshTags": tags,
            "meshFormat": "aflr3",
            "meshEndianness": endianness.value,
            "meshParams": params.json(),
        }

        if solver_version:
            body["solverVersion"] = solver_version

        resp = http.post("volumemeshes", body)
        if resp:
            return cls(**resp)
        return None
