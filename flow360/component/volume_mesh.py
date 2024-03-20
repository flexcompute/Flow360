"""
Volume mesh component
"""

from __future__ import annotations

import os.path
from enum import Enum
from typing import Iterator, List, Optional, Union

import numpy as np
from pydantic import Extra, Field, validator

from flow360.component.compress_upload import compress_and_upload_chunks

from ..cloud.requests import CopyExampleVolumeMeshRequest, NewVolumeMeshRequest
from ..cloud.rest_api import RestApi
from ..exceptions import (
    Flow360CloudFileError,
    Flow360FileError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
    Flow360ValueError,
)
from ..log import log
from ..solver_version import Flow360Version
from .case import Case, CaseDraft
from .flow360_params.boundaries import NoSlipWall
from .flow360_params.flow360_params import (
    Flow360MeshParams,
    Flow360Params,
    _GenericBoundaryWrapper,
)
from .flow360_params.params_base import params_generic_validator
from .interfaces import VolumeMeshInterface
from .meshing.params import VolumeMeshingParams
from .resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    Flow360ResourceListBase,
    ResourceDraft,
)
from .types import COMMENTS
from .utils import shared_account_confirm_proceed, validate_type, zstd_compress
from .validator import Validator

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
            for wall_name, wall in params.boundaries.dict().items()
            if wall_name != COMMENTS and _GenericBoundaryWrapper(v=wall).v.type == NoSlipWall().type
        ]

    return []


def get_boundaries_from_sliding_interfaces(params: Union[Flow360Params, Flow360MeshParams]):
    """
    Get wall boundary names
    :param params:
    :return:
    """
    assert params
    res = []

    # Sliding interfaces are deprecated - we need to handle this somehow
    # if params.sliding_interfaces and params.sliding_interfaces.rotating_patches:
    #    res += params.sliding_interfaces.rotating_patches[:]
    # if params.sliding_interfaces and params.sliding_interfaces.stationary_patches:
    #    res += params.sliding_interfaces.stationary_patches[:]
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
    boundaries_in_params = get_no_slip_walls(params) + get_boundaries_from_sliding_interfaces(
        params
    )
    boundaries_in_file = set(boundaries_in_file)
    boundaries_in_params = set(boundaries_in_params)

    if not boundaries_in_file.issuperset(boundaries_in_params):
        raise Flow360ValueError(
            "The following input boundary names from mesh json are not found in mesh:"
            + f" {' '.join(boundaries_in_params - boundaries_in_file)}."
            + f" Boundary names in cgns: {' '.join(boundaries_in_file)}"
            + f" Boundary names in params: {' '.join(boundaries_in_file)}"
        )
    log.info(
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
        raise Flow360RuntimeError(f"Unsupported file format {file}")


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
        raise Flow360RuntimeError(f"Unknown endianness for file {file}")


class CompressionFormat(Enum):
    """
    Volume mesh file format
    """

    GZ = "gz"
    BZ2 = "bz2"
    ZST = "zst"
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
        if self is CompressionFormat.ZST:
            return ".zst"
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
        if ext == CompressionFormat.ZST.ext():
            return CompressionFormat.ZST, file_name
        return CompressionFormat.NONE, file


# pylint: disable=E0213
class VolumeMeshMeta(Flow360ResourceBaseModel, extra=Extra.allow):
    """
    VolumeMeshMeta component
    """

    id: str = Field(alias="meshId")
    name: str = Field(alias="meshName")
    created_at: str = Field(alias="meshAddTime")
    surface_mesh_id: Optional[str] = Field(alias="surfaceMeshId")
    mesh_params: Union[Flow360MeshParams, None, dict] = Field(alias="meshParams")
    mesh_format: Union[VolumeMeshFileFormat, None] = Field(alias="meshFormat")
    file_name: Union[str, None] = Field(alias="fileName")
    endianness: UGRIDEndianness = Field(alias="meshEndianness")
    compression: CompressionFormat = Field(alias="meshCompression")
    boundaries: Union[List, None]

    @validator("mesh_params", pre=True)
    def init_mesh_params(cls, value):
        """
        validator for mesh_params
        """
        return params_generic_validator(value, Flow360MeshParams)

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


class VolumeMeshDraft(ResourceDraft):
    """
    Volume mesh draft component (before submit)
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        file_name: str = None,
        params: Union[Flow360MeshParams, VolumeMeshingParams] = None,
        name: str = None,
        surface_mesh_id=None,
        tags: List[str] = None,
        solver_version=None,
        endianess: UGRIDEndianness = None,
        isascii: bool = False,
    ):
        if file_name is not None and not os.path.exists(file_name):
            raise Flow360FileError(f"File '{file_name}' not found.")

        if endianess is not None:
            raise Flow360NotImplementedError(
                "endianess selections not supported, it is inferred from filename"
            )

        if isascii is True:
            raise Flow360NotImplementedError("isascii not supported")

        self.params = None
        if params is not None:
            if not isinstance(params, Flow360MeshParams) and not isinstance(
                params, VolumeMeshingParams
            ):
                raise ValueError(
                    f"params={params} are not of type Flow360MeshParams OR VolumeMeshingParams"
                )
            self.params = params.copy(deep=True)

        if name is None and file_name is not None:
            name = os.path.splitext(os.path.basename(file_name))[0]

        self.file_name = file_name
        self.name = name
        self.surface_mesh_id = surface_mesh_id
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self.compress_method = CompressionFormat.ZST
        ResourceDraft.__init__(self)

    def _submit_from_surface(self):
        self.validator_api(self.params, solver_version=self.solver_version)
        body = {
            "name": self.name,
            "tags": self.tags,
            "surfaceMeshId": self.surface_mesh_id,
            "config": self.params.flow360_json(),
            "format": "cgns",
        }

        if self.solver_version:
            body["solverVersion"] = self.solver_version

        resp = RestApi(VolumeMeshInterface.endpoint).post(body)
        if not resp:
            return None

        info = VolumeMeshMeta(**resp)
        self._id = info.id
        mesh = VolumeMesh(self.id)
        log.info(f"VolumeMesh successfully submitted: {mesh.short_description()}")
        return mesh

    # pylint: disable=protected-access, too-many-locals
    def _submit_upload_mesh(self, progress_callback=None):
        assert os.path.exists(self.file_name)

        original_compression, file_name_no_compression = CompressionFormat.detect(self.file_name)
        mesh_format = VolumeMeshFileFormat.detect(file_name_no_compression)
        endianness = UGRIDEndianness.detect(file_name_no_compression)
        if mesh_format is VolumeMeshFileFormat.CGNS:
            remote_file_name = "volumeMesh"
        else:
            remote_file_name = "mesh"
        compression = (
            original_compression
            if original_compression != CompressionFormat.NONE
            else self.compress_method
        )
        remote_file_name = (
            f"{remote_file_name}{endianness.ext()}{mesh_format.ext()}{compression.ext()}"
        )

        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.file_name))[0]

        req = NewVolumeMeshRequest(
            name=name,
            file_name=remote_file_name,
            tags=self.tags,
            format=mesh_format.value,
            endianness=endianness.value,
            compression=compression.value,
            params=self.params,
            solver_version=self.solver_version,
        )
        resp = RestApi(VolumeMeshInterface.endpoint).post(req.dict())
        if not resp:
            return None

        info = VolumeMeshMeta(**resp)
        self._id = info.id
        mesh = VolumeMesh(self.id)

        # parallel compress and upload
        if (
            original_compression == CompressionFormat.NONE
            and self.compress_method == CompressionFormat.BZ2
        ):
            upload_id = mesh.create_multipart_upload(remote_file_name)
            compress_and_upload_chunks(self.file_name, upload_id, mesh, remote_file_name)

        elif (
            original_compression == CompressionFormat.NONE
            and self.compress_method == CompressionFormat.ZST
        ):
            compressed_file_name = zstd_compress(self.file_name)
            mesh._upload_file(
                remote_file_name, compressed_file_name, progress_callback=progress_callback
            )
            os.remove(compressed_file_name)
        else:
            mesh._upload_file(remote_file_name, self.file_name, progress_callback=progress_callback)
        mesh._complete_upload(remote_file_name)

        log.info(f"VolumeMesh successfully uploaded: {mesh.short_description()}")
        return mesh

    def submit(self, progress_callback=None) -> VolumeMesh:
        """submit mesh to cloud

        Parameters
        ----------
        progress_callback : callback, optional
            Use for custom progress bar, by default None

        Returns
        -------
        VolumeMesh
            VolumeMesh object with id
        """

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        if self.file_name is not None:
            return self._submit_upload_mesh(progress_callback)

        if self.surface_mesh_id is not None and self.name is not None and self.params is not None:
            return self._submit_from_surface()

        raise Flow360ValueError(
            "You must provide volume mesh file for upload or surface mesh Id with meshing parameters."
        )

    @classmethod
    def validator_api(cls, params: VolumeMeshingParams, solver_version=None):
        """
        validation api: validates surface meshing parameters before submitting
        """
        return Validator.VOLUME_MESH.validate(params, solver_version=solver_version)


class VolumeMesh(Flow360Resource):
    """
    Volume mesh component
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=VolumeMeshInterface,
            info_type_class=VolumeMeshMeta,
            id=id,
        )
        self.__mesh_params = None

    @classmethod
    def _from_meta(cls, meta: VolumeMeshMeta):
        validate_type(meta, "meta", VolumeMeshMeta)
        volume_mesh = cls(id=meta.id)
        volume_mesh._set_meta(meta)
        return volume_mesh

    @property
    def info(self) -> VolumeMeshMeta:
        return super().info

    @property
    def _mesh_params(self) -> Flow360MeshParams:
        """
        returns mesh params
        """
        if self.__mesh_params is None:
            self.__mesh_params = self.info.mesh_params
        return self.__mesh_params

    @property
    def no_slip_walls(self):
        """
        returns mesh no_slip_walls
        """
        if self._mesh_params is None:
            return None
        return self._mesh_params.boundaries.no_slip_walls

    @property
    def all_boundaries(self):
        """
        returns mesh no_slip_walls
        """
        return self.info.boundaries

    # pylint: disable=too-many-arguments,R0801
    def download_file(
        self,
        file_name: Union[str, VolumeMeshDownloadable],
        to_file=None,
        to_folder=".",
        overwrite: bool = True,
        progress_callback=None,
        **kwargs,
    ):
        """
        Download file from surface mesh
        :param file_name:
        :param to_file:
        :return:
        """
        if isinstance(file_name, VolumeMeshDownloadable):
            file_name = file_name.value
        return super()._download_file(
            file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
            progress_callback=progress_callback,
            **kwargs,
        )

    # pylint: disable=R0801
    def download(self, to_file=None, to_folder=".", overwrite: bool = True):
        """
        Download volume mesh file
        :param to_file:
        :return:
        """
        status = self.status
        if not status.is_final():
            log.warning(f"Cannot download file because status={status}")
            return None

        remote_file_name = self.info.file_name
        if remote_file_name is None:
            remote_file_name = self._remote_file_name()

        return super()._download_file(
            remote_file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
        )

    def _complete_upload(self, remote_file_name):
        """
        Complete volume mesh upload
        :return:
        """
        resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
        self._info = VolumeMeshMeta(**resp)

    @classmethod
    def _interface(cls):
        return VolumeMeshInterface

    @classmethod
    def _meta_class(cls):
        """
        returns volume mesh meta info class: VolumeMeshMeta
        """
        return VolumeMeshMeta

    @classmethod
    def _params_ancestor_id_name(cls):
        """
        returns surfaceMeshId name
        """
        return "surfaceMeshId"

    @classmethod
    def from_cloud(cls, mesh_id: str):
        """
        Get volume mesh info from cloud
        :param mesh_id:
        :return:
        """
        return cls(mesh_id)

    def _get_file_extention(self):
        compression = self.info.compression
        mesh_format = self.info.mesh_format
        endianness = self.info.endianness
        return f"{endianness.ext()}{mesh_format.ext()}{compression.ext()}"

    def _remote_file_name(self):
        """
        mesh filename on cloud
        """

        remote_file_name = None
        for file in self.get_download_file_list():
            _, file_name_no_compression = CompressionFormat.detect(file["fileName"])
            try:
                VolumeMeshFileFormat.detect(file_name_no_compression)
                remote_file_name = file["fileName"]
            except Flow360RuntimeError:
                continue

        if remote_file_name is None:
            raise Flow360CloudFileError(f"No volume mesh file found for id={self.id}")

        return remote_file_name

    @classmethod
    def from_file(
        cls,
        file_name: str,
        params: Union[Flow360MeshParams, None] = None,
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
        endianess: UGRIDEndianness = None,
        isascii: bool = False,
    ) -> VolumeMeshDraft:
        """
        Upload volume mesh from file
        :param volume_mesh_name:
        :param file_name:
        :param params:
        :param tags:
        :param solver_version:
        :return:
        """
        return VolumeMeshDraft(
            file_name=file_name,
            name=name,
            tags=tags,
            solver_version=solver_version,
            params=params,
            endianess=endianess,
            isascii=isascii,
        )

    @classmethod
    def copy_from_example(
        cls,
        example_id: str,
        name: str = None,
    ) -> VolumeMesh:
        """
        Create a new volume mesh by copying from an example mesh identified by `example_id`.

        Parameters
        ----------
        example_id : str
            The unique identifier of the example volume mesh to copy from.
        name : str, optional
            The name to assign to the new volume mesh. If not provided, the name
            of the example volume mesh will be used.

        Returns
        -------
        VolumeMesh
            A new instance of VolumeMesh copied from the example mesh if successful.

        Examples
        --------
        >>> new_mesh = VolumeMesh.copy_from_example('example_id_123', name='New Mesh')
        """

        if name is None:
            eg_vm = cls(example_id)
            name = eg_vm.name
        req = CopyExampleVolumeMeshRequest(example_id=example_id, name=name)
        resp = RestApi(f"{VolumeMeshInterface.endpoint}/examples/copy").post(req.dict())
        if not resp:
            raise RuntimeError("Something went wrong when accessing example mesh.")

        info = VolumeMeshMeta(**resp)
        return cls(info.id)

    @classmethod
    def create(
        cls,
        name: str,
        params: VolumeMeshingParams,
        surface_mesh_id,
        tags: List[str] = None,
        solver_version=None,
    ) -> VolumeMeshDraft:
        """
        Create volume mesh from surface mesh
        """

        return VolumeMeshDraft(
            name=name,
            surface_mesh_id=surface_mesh_id,
            solver_version=solver_version,
            params=params,
            tags=tags,
        )

    def create_case(
        self,
        name: str,
        params: Flow360Params,
        tags: List[str] = None,
        solver_version: str = None,
    ) -> CaseDraft:
        """
        Create new case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return Case.create(
            name, params, volume_mesh_id=self.id, tags=tags, solver_version=solver_version
        )


class VolumeMeshList(Flow360ResourceListBase):
    """
    VolumeMesh List component
    """

    def __init__(
        self,
        surface_mesh_id: str = None,
        from_cloud: bool = True,
        include_deleted: bool = False,
        limit=100,
    ):
        super().__init__(
            ancestor_id=surface_mesh_id,
            from_cloud=from_cloud,
            include_deleted=include_deleted,
            limit=limit,
            resourceClass=VolumeMesh,
        )

    def filter(self):
        """
        flitering list, not implemented yet
        """
        raise NotImplementedError("Filters are not implemented yet")

        # resp = list(filter(lambda i: i['caseStatus'] != 'deleted', resp))

    # pylint: disable=useless-parent-delegation
    def __getitem__(self, index) -> VolumeMesh:
        """
        returns VolumeMeshMeta item of the list
        """
        return super().__getitem__(index)

    # pylint: disable=useless-parent-delegation
    def __iter__(self) -> Iterator[VolumeMesh]:
        return super().__iter__()
