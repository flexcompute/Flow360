"""
Volume mesh component
"""
from __future__ import annotations
import os.path
import json
from enum import Enum
from typing import Optional, Union, List

import numpy as np
from pydantic import Extra, Field, validator, ValidationError

from ..cloud.s3_utils import S3TransferType
from ..cloud.rest_api import RestApi
from ..solver_version import Flow360Version
from .flow360_base_model import (
    Flow360BaseModel,
    Flow360Resource,
    on_cloud_resource_only,
)
from .flow360_solver_params import (
    Flow360MeshParams,
    Flow360Params,
    MeshBoundary,
    NoSlipWall,
)

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


class VolumeMeshFileFormat(Enum):
    """
    Volume mesh file format
    """

    UGRID = "aflr3"
    CGNS = "cgns"

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is VolumeMeshFileFormat.UGRID:
            return ".ugrid"
        if self is VolumeMeshFileFormat.CGNS:
            return ".cgns"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects mesh format from filename
        """
        ext = os.path.splitext(file)[1]
        if ext == VolumeMeshFileFormat.UGRID.ext():
            return VolumeMeshFileFormat.UGRID
        if ext == VolumeMeshFileFormat.CGNS.ext():
            return VolumeMeshFileFormat.CGNS
        raise RuntimeError(f"Unsupported file format {file}")


class UGRIDEndianness(Enum):
    """
    UGRID endianness
    """

    LITTLE = "little"
    BIG = "big"
    NONE = None

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is UGRIDEndianness.LITTLE:
            return ".lb8"
        if self is UGRIDEndianness.BIG:
            return ".b8"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects endianess UGRID mesh from filename
        """
        if VolumeMeshFileFormat.detect(file) is not VolumeMeshFileFormat.UGRID:
            return UGRIDEndianness.NONE
        basename = os.path.splitext(file)[0]
        ext = os.path.splitext(basename)[1]
        if ext == UGRIDEndianness.LITTLE.ext():
            return UGRIDEndianness.LITTLE
        if ext == UGRIDEndianness.BIG.ext():
            return UGRIDEndianness.BIG
        raise RuntimeError(f"Unknown endianness for file {file}")


class CompressionFormat(Enum):
    """
    Volume mesh file format
    """

    GZ = "gz"
    BZ2 = "bz2"
    NONE = None

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is CompressionFormat.GZ:
            return ".gz"
        if self is CompressionFormat.BZ2:
            return ".bz2"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects compression from filename
        """
        file_name, ext = os.path.splitext(file)
        if ext == CompressionFormat.GZ.ext():
            return CompressionFormat.GZ, file_name
        if ext == CompressionFormat.BZ2.ext():
            return CompressionFormat.BZ2, file_name
        return CompressionFormat.NONE, file


# pylint: disable=E0213
class VolumeMeshMeta(Flow360BaseModel, extra=Extra.allow):
    """
    VolumeMeshMeta component
    """

    id: str = Field(alias="meshId")
    name: str = Field(alias="meshName")
    status: str = Field(alias="meshStatus")
    created_at: str = Field(alias="meshAddTime")
    surface_mesh_id: Optional[str] = Field(alias="surfaceMeshId")
    mesh_params: Union[Flow360MeshParams, None, dict] = Field(alias="meshParams")
    mesh_format: VolumeMeshFileFormat = Field(alias="meshFormat")
    endianness: UGRIDEndianness = Field(alias="meshEndianness")
    compression: CompressionFormat = Field(alias="meshCompression")
    boundaries: Union[List, None]

    @validator("mesh_params", pre=True)
    def init_mesh_params(cls, value):
        """
        validator for mesh_params
        """
        params = value
        if isinstance(value, str):
            try:
                params = json.loads(value)
            except json.decoder.JSONDecodeError:
                return None
        try:
            Flow360MeshParams(**params)
        except ValidationError:
            return None
        except TypeError:
            return None

        return params

    @validator("endianness", pre=True)
    def init_endianness(cls, value):
        """
        validator for endianess
        """
        return UGRIDEndianness(value) or UGRIDEndianness.NONE

    @validator("compression", pre=True)
    def init_compression(cls, value):
        """
        validator for compression
        """
        try:
            return CompressionFormat(value)
        except ValueError:
            return CompressionFormat.NONE

    def to_volume_mesh(self) -> VolumeMesh:
        """
        returns VolumeMesh object from volume mesh meta info
        """
        return VolumeMesh(self.id)


class VolumeMesh(Flow360Resource):
    """
    Volume mesh component
    """

    def __init__(self, mesh_id: str = None):
        super().__init__(
            resource_type="Volume Mesh",
            info_type_class=VolumeMeshMeta,
            s3_transfer_method=S3TransferType.VOLUME_MESH,
            endpoint="volumemeshes",
            id=mesh_id,
        )
        if mesh_id is not None:
            self.get_info()
            try:
                self._params = Flow360MeshParams(**self.info.mesh_params.dict())
            except AttributeError:
                self._params = Flow360MeshParams(boundaries=MeshBoundary(no_slip_walls=[]))

    @property
    def info(self) -> VolumeMeshMeta:
        return super().info

    @property
    @on_cloud_resource_only
    def params(self) -> Flow360MeshParams:
        """
        returns mesh params
        """
        return self._params

    @property
    @on_cloud_resource_only
    def no_slip_walls(self):
        """
        returns mesh no_slip_walls
        """
        return self._params.boundaries.no_slip_walls

    @property
    @on_cloud_resource_only
    def all_boundaries(self):
        """
        returns mesh no_slip_walls
        """
        return self.info.boundaries

    # pylint: disable=too-many-arguments
    @on_cloud_resource_only
    def download_file(
        self,
        file_name: Union[str, VolumeMeshDownloadable],
        to_file=".",
        keep_folder: bool = True,
        overwrite: bool = True,
        progress_callback=None,
    ):
        """
        Download file from surface mesh
        :param file_name:
        :param to_file:
        :param keep_folder:
        :return:
        """
        if isinstance(file_name, VolumeMeshDownloadable):
            file_name = file_name.value
        return super().download_file(
            file_name,
            to_file,
            keep_folder=keep_folder,
            overwrite=overwrite,
            progress_callback=progress_callback,
        )

    @on_cloud_resource_only
    def download(self, to_file=".", keep_folder: bool = True):
        """
        Download volume mesh file
        :param to_file:
        :param keep_folder:
        :return:
        """
        super().download_file(self._remote_file_name(), to_file, keep_folder)

    @on_cloud_resource_only
    def download_log(self, log: VolumeMeshLog, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param log:
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download_file(f"logs/{log.value}", to_file, keep_folder)

    @on_cloud_resource_only
    def _complete_upload(self, remote_file_name):
        """
        Complete volume mesh upload
        :return:
        """
        resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
        self._info = VolumeMeshMeta(**resp)

    @classmethod
    def from_cloud(cls, mesh_id: str):
        """
        Get volume mesh info from cloud
        :param mesh_id:
        :return:
        """
        return cls(mesh_id)

    # pylint: disable=too-many-arguments
    # @classmethod
    # def from_surface_mesh(
    #     cls,
    #     volume_mesh_name: str,
    #     surface_mesh_id: str,
    #     config_file: str,
    #     tags: [str] = None,
    #     solver_version=None,
    # ):
    #     """
    #     Create volume mesh from surface mesh
    #     :param volume_mesh_name:
    #     :param surface_mesh_id:
    #     :param config_file:
    #     :param tags:
    #     :param solver_version:
    #     :return:
    #     """
    #     assert volume_mesh_name
    #     assert os.path.exists(config_file)
    #     assert surface_mesh_id
    #     with open(config_file, "r", encoding="utf-8") as config_f:
    #         json_content = json.load(config_f)
    #     body = {
    #         "name": volume_mesh_name,
    #         "tags": tags,
    #         "surfaceMeshId": surface_mesh_id,
    #         "config": json.dumps(json_content),
    #         "format": "cgns",
    #     }

    #     if solver_version:
    #         body["solverVersion"] = solver_version

    #     resp = http.post("volumemeshes", body)
    #     if resp:
    #         return cls(**resp)
    #     return None

    @on_cloud_resource_only
    def _remote_file_name(self, mesh_format=None, compression=None, endianness=None):
        """
        mesh filename on cloud
        """
        compression = compression or self.info.compression
        mesh_format = mesh_format or self.info.mesh_format
        endianness = endianness or self.info.endianness

        remote_file_name = "mesh"
        if mesh_format is VolumeMeshFileFormat.CGNS:
            remote_file_name = self.info.name

        return f"{remote_file_name}{endianness.ext()}{mesh_format.ext()}{compression.ext()}"

    @classmethod
    def from_file(
        cls,
        file_name: str,
        params: Flow360MeshParams,
        name: str = None,
        tags: [str] = None,
        solver_version=None,
        progress_callback=None,
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
        assert os.path.exists(file_name)
        assert params

        if name is None:
            name = os.path.splitext(os.path.basename(file_name))[0]

        mesh = cls()
        compression, file_name_no_compression = CompressionFormat.detect(file_name)
        mesh_format = VolumeMeshFileFormat.detect(file_name_no_compression)
        endianness = UGRIDEndianness.detect(file_name_no_compression)

        body = {
            "meshName": name,
            "meshTags": tags,
            "meshFormat": mesh_format.value,
            "meshEndianness": endianness.value,
            "meshParams": params.json(),
        }

        if solver_version:
            body["solverVersion"] = solver_version

        resp = mesh.post(body)

        if not resp:
            return None

        mesh._info = VolumeMeshMeta(**resp)
        mesh._params = Flow360MeshParams(**mesh._info.mesh_params.dict())
        mesh.init_id(mesh._info.id)
        remote_file_name = mesh._remote_file_name(mesh_format, compression, endianness)
        mesh.upload_file(remote_file_name, file_name, progress_callback=progress_callback)
        mesh._complete_upload(remote_file_name)
        return mesh

    #     # pylint: disable=too-many-arguments

    # @classmethod
    # def from_cgns_file(
    #     cls,
    #     volume_mesh_name: str,
    #     file_name: str,
    #     params: Union[Flow360Params, Flow360MeshParams],
    #     tags: [str] = None,
    #     solver_version=None,
    # ):
    #     """
    #     Create volume mesh from ugrid file
    #     :param volume_mesh_name:
    #     :param file_name:
    #     :param params:
    #     :param tags:
    #     :param solver_version:
    #     :return:
    #     """
    #     assert volume_mesh_name
    #     assert os.path.exists(file_name)
    #     assert params

    #     if _H5PY_AVAILABLE:
    #         validate_cgns(file_name, params, solver_version=solver_version)
    #     else:
    #         warnings.warn(
    #             "Could not check consistency between mesh file and"
    #             " Flow360.json file. h5py module not found. This is optional functionality"
    #         )

    #     body = {
    #         "meshName": volume_mesh_name,
    #         "meshTags": tags,
    #         "meshFormat": "cgns",
    #         "meshParams": params.json(),
    #     }

    #     if solver_version:
    #         body["solverVersion"] = solver_version

    #     resp = http.post("volumemeshes", body)
    #     if resp:
    #         return cls(**resp)
    #     return None


class VolumeMeshList(list, RestApi):
    """
    VolumeMesh List component
    """

    def __init__(self, from_cloud: bool = True, include_deleted: bool = False):
        RestApi.__init__(self, endpoint="volumemeshes")

        if from_cloud:
            resp = self.get(params={"includeDeleted": include_deleted})
            list.__init__(self, [VolumeMeshMeta(**item) for item in resp])

    def filter(self):
        """
        flitering list, not implemented yet
        """
        # resp = list(filter(lambda i: i['caseStatus'] != 'deleted', resp))

    def __getitem__(self, index) -> VolumeMeshMeta:
        """
        returns VolumeMeshMeta item of the list
        """
        return super().__getitem__(index)

    @classmethod
    def from_cloud(cls):
        """
        get VolumeMeshList from cloud
        """
        return cls(from_cloud=True)
