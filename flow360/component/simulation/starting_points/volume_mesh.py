"""
Volume mesh component for simulation strucure
"""

from __future__ import annotations

import os.path
from enum import Enum
from typing import Iterator, List, Optional, Union

import numpy as np
from pydantic import Field, field_validator

from flow360.cloud.requests import CopyExampleVolumeMeshRequest, NewVolumeMeshRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case, CaseDraft
from flow360.component.compress_upload import compress_and_upload_chunks

## TODO: flow360.component.flow360_params reimplementation
from flow360.component.flow360_params.boundaries import NoSlipWall
from flow360.component.flow360_params.params_base import params_generic_validator
from flow360.component.interfaces import VolumeMeshInterface
from flow360.component.meshing.params import VolumeMeshingParams
from flow360.component.simulation.meshing_param.params import Flow360MeshParams
from flow360.component.types import COMMENTS
from flow360.component.utils import (
    shared_account_confirm_proceed,
    validate_type,
    zstd_compress,
)
from flow360.component.validator import Validator
from flow360.component_v2.resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    Flow360ResourceListBase,
    ResourceDraft,
)
from flow360.exceptions import (
    Flow360CloudFileError,
    Flow360FileError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
    Flow360ValueError,
)
from flow360.log import log


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
class VolumeMeshMeta(Flow360ResourceBaseModel):
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

    @field_validator("mesh_params", mode="before")
    def init_mesh_params(cls, value):
        """
        validator for mesh_params
        """
        return params_generic_validator(value, Flow360MeshParams)

    @field_validator("endianness", mode="before")
    def init_endianness(cls, value):
        """
        validator for endianess
        """
        return UGRIDEndianness(value) or UGRIDEndianness.NONE

    @field_validator("compression", mode="before")
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
        return VolumeMesh(id=self.id)


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

    def _submit_from_surface(self, force_submit: bool = False):
        self.validator_api(
            self.params, solver_version=self.solver_version, raise_on_error=(not force_submit)
        )
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
        mesh = VolumeMesh(id=self.id, file_name=None)

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

    def submit(self, progress_callback=None, force_submit: bool = False) -> VolumeMesh:
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
            return self._submit_from_surface(force_submit=force_submit)

        raise Flow360ValueError(
            "You must provide volume mesh file for upload or surface mesh Id with meshing parameters."
        )

    @classmethod
    def validator_api(
        cls, params: VolumeMeshingParams, solver_version=None, raise_on_error: bool = True
    ):
        """
        validation api: validates surface meshing parameters before submitting
        """
        return Validator.VOLUME_MESH.validate(
            params, solver_version=solver_version, raise_on_error=raise_on_error
        )


class VolumeMesh(Flow360Resource):
    """
    Volume mesh component
    """

    volume_mesh_draft: Optional[VolumeMeshDraft] = None

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        id: str = None,
        file_name: str = None,
        params: Union[Flow360MeshParams, None] = None,
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
        endianess: UGRIDEndianness = None,
        isascii: bool = False,
    ):
        if id is not None:
            assert file_name is None
            super().__init__(
                interface=VolumeMeshInterface,
                info_type_class=VolumeMeshMeta,
                id=id,
            )
        else:
            self.volume_mesh_draft = VolumeMeshDraft(
                file_name=file_name,
                name=name,
                tags=tags,
                solver_version=solver_version,
                params=params,
                endianess=endianess,
                isascii=isascii,
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
        return cls(id=mesh_id)

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
    ):
        """
        Prepare uploading volume mesh from file
        """

        return cls(
            id=None,
            file_name=file_name,
            name=name,
            tags=tags,
            solver_version=solver_version,
            params=params,
            endianess=endianess,
            isascii=isascii,
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
