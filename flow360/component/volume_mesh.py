"""
Surface mesh component
"""
import json
import os.path
import warnings
from enum import Enum
from typing import Optional, Union

import numpy as np
from pydantic import Extra, Field

from flow360.cloud.http_util import http
from flow360.cloud.s3_utils import S3TransferType
from flow360.component.flow360_base_model import Flow360BaseModel
from flow360.component.flow360_solver_params import (
    Flow360MeshParams,
    Flow360Params,
    NoSlipWall,
)
from flow360.version import Flow360Version

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def get_datatype(dataset):
    """
    Get datatype of dataset
    :param dataset:
    :return:
    """
    data_raw = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(data_raw)
    data_str = "".join([chr(i) for i in dataset])
    return data_str


def get_no_slip_walls(params: Union[Flow360Params, Flow360MeshParams]):
    """
    Get wall boundary names
    :param params:
    :param solver_version:
    :return:
    """
    assert params
    if (
        isinstance(params, Flow360MeshParams)
        and params.boundaries
        and params.boundaries.no_slip_walls
    ):
        return params.boundaries.no_slip_walls

    if isinstance(params, Flow360Params) and params.boundaries:
        return [
            wall_name
            for wall_name, wall in params.boundaries.items()
            if isinstance(wall, NoSlipWall)
        ]

    return []


def get_boundries_from_sliding_interfaces(params: Union[Flow360Params, Flow360MeshParams]):
    """
    Get wall boundary names
    :param params:
    :param solver_version:
    :return:
    """
    assert params
    res = []

    if params.sliding_interfaces and params.sliding_interfaces.rotating_patches:
        res += params.sliding_interfaces.rotating_patches[:]
    if params.sliding_interfaces and params.sliding_interfaces.stationary_patches:
        res += params.sliding_interfaces.stationary_patches[:]
    return res


# pylint: disable=too-many-branches
def get_boundaries_from_file(cgns_file: str, solver_version: str = None):
    """
    Get boundary names from CGNS file
    :param cgns_file:
    :param solver_version:
    :return:
    """
    names = []
    with h5py.File(cgns_file, "r") as h5_file:
        base = h5_file["Base"]
        for zone_name, zone in base.items():
            if zone_name == " data":
                continue
            if zone.attrs["label"].decode() != "Zone_t":
                continue
            zone_type = get_datatype(base[f"{zone_name}/ZoneType/ data"])
            if zone_type not in ["Structured", "Unstructured"]:
                continue
            for section_name, section in zone.items():
                if section_name == " data":
                    continue
                if "label" not in section.attrs:
                    continue
                if solver_version and Flow360Version(solver_version) < Flow360Version(
                    "release-22.2.1.0"
                ):
                    if section.attrs["label"].decode() != "Elements_t":
                        continue
                    element_type_tag = int(zone[f"{section_name}/ data"][0])
                    if element_type_tag in [5, 7]:
                        names.append(f"{zone_name}/{section_name}")
                    if element_type_tag == 20:
                        first_element_type_tag = zone[f"{section_name}/ElementConnectivity/ data"][
                            0
                        ]
                        if first_element_type_tag in [5, 7]:
                            names.append(f"{zone_name}/{section_name}")
                else:
                    if section.attrs["label"].decode() != "ZoneBC_t":
                        continue
                    for bc_name, bc_zone in section.items():

                        if bc_zone.attrs["label"].decode() == "BC_t":
                            names.append(f"{zone_name}/{bc_name}")

        return names


def validate_cgns(
    cgns_file: str, params: Union[Flow360Params, Flow360MeshParams], solver_version=None
):
    """
    Validate CGNS file
    :param cgns_file:
    :param params:
    :param solver_version:
    :return:
    """
    assert cgns_file
    assert params
    boundaries_in_file = get_boundaries_from_file(cgns_file, solver_version)

    boundaries_in_params = get_no_slip_walls(params) + get_boundries_from_sliding_interfaces(params)
    boundaries_in_file = set(boundaries_in_file)
    boundaries_in_params = set(boundaries_in_params)
    if not boundaries_in_file.issuperset(boundaries_in_params):
        raise ValueError(
            "The following input boundary names from mesh json are not found in mesh:"
            + f" {' '.join(boundaries_in_params - boundaries_in_file)}."
            + f" Boundary names in cgns: {' '.join(boundaries_in_file)}"
            + f" Boundary names in params: {' '.join(boundaries_in_file)}"
        )
    print(
        f'Notice: {" ".join(boundaries_in_file - boundaries_in_params)} is '
        + "tagged as wall in cgns file, but not in input params"
    )


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

        # pylint: disable=too-many-arguments

    @classmethod
    def from_cgns_file(
        cls,
        volume_mesh_name: str,
        file_name: str,
        params: Union[Flow360Params, Flow360MeshParams],
        tags: [str] = None,
        solver_version=None,
    ):
        """
        Create volume mesh from ugrid file
        :param volume_mesh_name:
        :param file_name:
        :param params:
        :param tags:
        :param solver_version:
        :return:
        """
        assert volume_mesh_name
        assert os.path.exists(file_name)
        assert params

        if _H5PY_AVAILABLE:
            validate_cgns(file_name, params, solver_version=solver_version)
        else:
            warnings.warn(
                "Could not check consistency between mesh file and"
                " Flow360.json file. h5py module not found. This is optional functionality"
            )

        body = {
            "meshName": volume_mesh_name,
            "meshTags": tags,
            "meshFormat": "aflr3",
            "meshParams": params.json(),
        }

        if solver_version:
            body["solverVersion"] = solver_version

        resp = http.post("volumemeshes", body)
        if resp:
            return cls(**resp)
        return None
