"""
Surface mesh component
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Iterator, List, Union

import pydantic.v1 as pd

from flow360.component.v1.cloud.flow360_requests import NewSurfaceMeshRequest
from flow360.component.v1.meshing.params import (
    SurfaceMeshingParams,
    VolumeMeshingParams,
)
from flow360.flags import Flags

from ..cloud.rest_api import RestApi
from ..exceptions import Flow360FileError, Flow360ValueError
from ..log import log
from .interfaces import SurfaceMeshInterface
from .resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    Flow360ResourceListBase,
    ResourceDraft,
)
from .utils import (
    CompressionFormat,
    MeshFileFormat,
    MeshNameParser,
    UGRIDEndianness,
    _local_download_overwrite,
    shared_account_confirm_proceed,
    validate_type,
    zstd_compress,
)
from .v1.params_base import params_generic_validator
from .validator import Validator
from .volume_mesh import VolumeMeshDraft

SURFACE_MESH_NAME_STEM_V1 = "surfaceMesh"
SURFACE_MESH_NAME_STEM_V2 = "surface_mesh"


class SurfaceMeshDownloadable(Enum):
    """
    Surface Mesh downloadable files
    """

    CONFIG_JSON = "config.json"


# pylint: disable=E0213
class SurfaceMeshMeta(AssetMetaBaseModel, extra=pd.Extra.allow):
    """
    SurfaceMeshMeta component
    """

    params: Union[SurfaceMeshingParams, None, dict] = pd.Field(alias="config")
    mesh_format: Union[MeshFileFormat, None]
    geometry_id: Union[str, None] = pd.Field(alias="geometryId")

    @pd.validator("params", pre=True)
    def init_params(cls, value):
        """
        validator for params
        """
        return params_generic_validator(value, SurfaceMeshingParams)

    def to_surface_mesh(self) -> SurfaceMesh:
        """
        returns VolumeMesh object from volume mesh meta info
        """
        return SurfaceMesh(self.id)


# pylint: disable=too-many-instance-attributes
class SurfaceMeshDraft(ResourceDraft):
    """
    Surface Mesh Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        geometry_id: str = None,
        surface_mesh_file: str = None,
        params: SurfaceMeshingParams = None,
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
    ):
        self._geometry_id = geometry_id
        self._surface_mesh_file = surface_mesh_file
        self.name = name
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self.params = params
        self.compress_method = CompressionFormat.ZST
        self._validate()
        if params is not None:
            self.params = params.copy(deep=True)
        ResourceDraft.__init__(self)

    def _validate(self):
        if self.geometry_id is not None:
            pass
        elif self.surface_mesh_file is not None:
            self._validate_surface_mesh()
        else:
            raise Flow360ValueError(
                "One of geometry_id and surface_mesh_file has to be given to create a surface mesh."
            )

    def _validate_surface_mesh(self):
        mesh_parser = MeshNameParser(self.surface_mesh_file)
        if not mesh_parser.is_valid_surface_mesh():
            raise Flow360ValueError(
                f"Unsupported surface mesh file extensions: {mesh_parser.format}. Supported: [stl,ugrid,cgns]."
            )

        if not os.path.exists(self.surface_mesh_file):
            raise Flow360FileError(f"{self.surface_mesh_file} not found.")

    @property
    def geometry_id(self) -> str:
        """geometry id"""
        return self._geometry_id

    @property
    def surface_mesh_file(self) -> str:
        """surface mesh file"""
        return self._surface_mesh_file

    # pylint: disable=protected-access
    def _submit_from_geometry_id(self, force_submit: bool = False):
        if not force_submit:
            self.validator_api(self.params, solver_version=self.solver_version)

        stem = SURFACE_MESH_NAME_STEM_V1
        if Flags.beta_features():
            if self.params is not None and self.params.version == "v2":
                stem = SURFACE_MESH_NAME_STEM_V2

        if Flags.beta_features():
            req = NewSurfaceMeshRequest(
                name=self.name,
                stem=stem,
                tags=self.tags,
                geometry_id=self.geometry_id,
                mesh_format=MeshFileFormat.UGRID.value,
                endianness=UGRIDEndianness.LITTLE.value,
                config=self.params.flow360_json(),
                solver_version=self.solver_version,
                version=self.params.version,
            )
        else:
            req = NewSurfaceMeshRequest(
                name=self.name,
                stem=stem,
                tags=self.tags,
                geometry_id=self.geometry_id,
                config=self.params.flow360_json(),
                solver_version=self.solver_version,
            )
        resp = RestApi(SurfaceMeshInterface.endpoint).post(req.dict())
        info = SurfaceMeshMeta(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        submitted_mesh = SurfaceMesh(self.id)
        log.info(f"SurfaceMesh successfully submitted: {submitted_mesh.short_description()}")
        return submitted_mesh

    # pylint: disable=protected-access, too-many-locals
    def _submit_upload_mesh(self, progress_callback=None):
        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.surface_mesh_file))[0]

        mesh_parser = MeshNameParser(self.surface_mesh_file)
        original_compression = mesh_parser.compression
        mesh_format = mesh_parser.format
        endianness = mesh_parser.endianness

        compression = (
            original_compression
            if original_compression != CompressionFormat.NONE
            else self.compress_method
        )

        req = NewSurfaceMeshRequest(
            name=name,
            stem=SURFACE_MESH_NAME_STEM_V2,
            tags=self.tags,
            mesh_format=mesh_format.value,
            endianness=endianness.value,
            compression=compression.value,
            solver_version=self.solver_version,
        )

        resp = RestApi(SurfaceMeshInterface.endpoint).post(req.dict())
        info = SurfaceMeshMeta(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        submitted_mesh = SurfaceMesh(self.id)

        remote_file_name = (
            f"{SURFACE_MESH_NAME_STEM_V2}{endianness.ext()}{mesh_format.ext()}{compression.ext()}"
        )

        # upload self.surface_mesh_file
        if (
            original_compression == CompressionFormat.NONE
            and self.compress_method == CompressionFormat.ZST
        ):
            compressed_file_name = zstd_compress(self.surface_mesh_file)
            submitted_mesh._upload_file(
                remote_file_name, compressed_file_name, progress_callback=progress_callback
            )
            os.remove(compressed_file_name)
        else:
            submitted_mesh._upload_file(
                remote_file_name, self.surface_mesh_file, progress_callback=progress_callback
            )
        submitted_mesh._complete_upload(remote_file_name)
        # upload mapbc file if it exists in the same directory
        if mesh_parser.is_ugrid() and os.path.isfile(mesh_parser.get_associated_mapbc_filename()):
            remote_mesh_parser = MeshNameParser(remote_file_name)
            submitted_mesh._upload_file(
                remote_mesh_parser.get_associated_mapbc_filename(),
                mesh_parser.get_associated_mapbc_filename(),
                progress_callback=progress_callback,
            )
            submitted_mesh._complete_upload(remote_mesh_parser.get_associated_mapbc_filename())
            log.info(
                f"The {mesh_parser.get_associated_mapbc_filename()} is found and successfully submitted"
            )
        log.info(f"SurfaceMesh successfully submitted: {submitted_mesh.short_description()}")
        return submitted_mesh

    # pylint: disable=protected-access, too-many-branches
    def submit(self, progress_callback=None, force_submit: bool = False) -> SurfaceMesh:
        """submit surface meshing to cloud

        Parameters
        ----------
        progress_callback : callback, optional
            Use for custom progress bar, by default None

        Returns
        -------
        SurfaceMesh
            SurfaceMesh object with id
        """

        self._validate()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        if self.geometry_id is not None:
            return self._submit_from_geometry_id(force_submit=force_submit)
        if self.surface_mesh_file is not None:
            return self._submit_upload_mesh(progress_callback)

        raise Flow360ValueError(
            "You must provide surface mesh file for upload or geometry Id or geometry file with meshing parameters."
        )

    @classmethod
    def validator_api(cls, params: SurfaceMeshingParams, solver_version=None):
        """
        validation api: validates surface meshing parameters before submitting
        """
        return Validator.SURFACE_MESH.validate(params, solver_version=solver_version)


class SurfaceMesh(Flow360Resource):
    """
    Surface mesh component
    """

    _cloud_resource_type_name = "SurfaceMesh"

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=SurfaceMeshInterface,
            meta_class=SurfaceMeshMeta,
            id=id,
        )
        self._params = None

    @classmethod
    def _from_meta(cls, meta: SurfaceMeshMeta):
        validate_type(meta, "meta", SurfaceMeshMeta)
        surface_mesh = cls(id=meta.id)
        surface_mesh._set_meta(meta)
        return surface_mesh

    @property
    def info(self) -> SurfaceMeshMeta:
        return super().info

    @property
    def params(self) -> SurfaceMeshingParams:
        """
        returns meshing params
        """
        if self._params is None:
            if self.info.params is None:
                self.get_info(force=True)
            self._params = self.info.params
        return self._params

    # pylint: disable=too-many-arguments
    # def download_file(
    #     self,
    #     file_name: Union[str, SurfaceMeshDownloadable],
    #     to_file=".",
    #     overwrite: bool = True,
    #     progress_callback=None,
    # ):
    #     """
    #     Download file from surface mesh
    #     :param file_name:
    #     :param to_file:
    #     :return:
    #     """
    #     if isinstance(file_name, SurfaceMeshDownloadable):
    #         file_name = file_name.value
    #     return super().download_file(
    #         file_name,
    #         to_file,
    #         overwrite=overwrite,
    #         progress_callback=progress_callback,
    #     )

    # def download(self, to_file="."):
    #     """
    #     Download surface mesh file
    #     :param to_file:
    #     :return:
    #     """
    #     super().download_file(, to_file)

    @classmethod
    def _interface(cls):
        return SurfaceMeshInterface

    @classmethod
    def _meta_class(cls):
        """
        returns surface mesh meta info class: SurfaceMeshMeta
        """
        return SurfaceMeshMeta

    # pylint: disable=arguments-differ
    def _complete_upload(self, remote_file_name):
        """
        Complete surface mesh upload
        :return:
        """
        resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
        self._info = SurfaceMeshMeta(**resp)

    @classmethod
    # pylint: disable=unused-argument
    def from_cloud(cls, surface_mesh_id: str, **kwargs):
        """
        Get surface mesh from cloud
        :param surface_mesh_id:
        :return:
        """
        return cls(surface_mesh_id)

    @classmethod
    def from_file(
        cls,
        surface_mesh_file: str,
        name: str = None,
        tags: List[str] = None,
    ):
        """
        Create surface mesh from surface mesh file
        :param surface_mesh_file:
        :param name:
        :param tags:
        :param solver_version:
        :return:
        """
        return SurfaceMeshDraft(
            surface_mesh_file=surface_mesh_file,
            name=name,
            tags=tags,
        )

    @classmethod
    def from_local_storage(cls, local_storage_path, meta_data: SurfaceMeshMeta) -> SurfaceMesh:
        """
        Create a `SurfaceMesh` instance from local storage.

        Parameters
        ----------
        local_storage_path : str
            The path to the local storage directory.
        meta_data : SurfaceMeshMeta
            surface mesh metadata such as:
            id : str
                The unique identifier for the case.
            name : str
                The name of the case.
            user_id : str
                The user ID associated with the case, can be "local".

        Returns
        -------
        SurfaceMesh
            An instance of `SurfaceMesh` with data loaded from local storage.
        """
        _local_download_file = _local_download_overwrite(local_storage_path, cls.__name__)
        case = cls._from_meta(meta_data)
        case._download_file = _local_download_file
        return case

    @classmethod
    def create(
        cls,
        name: str,
        params: SurfaceMeshingParams,
        geometry_id: str,
        tags: List[str] = None,
        solver_version: str = None,
    ) -> SurfaceMeshDraft:
        """ "Create new surface mesh from geometry"

        Parameters
        ----------
        name : str
            _description_
        params : SurfaceMeshingParams
            _description_
        geometry_id : str
            _description_
        tags : List[str], optional
            _description_, by default None
        solver_version : str, optional
            _description_, by default None

        Returns
        -------
        SurfaceMeshDraft
            _description_
        """
        new_surface_mesh = SurfaceMeshDraft(
            geometry_id=geometry_id,
            params=params,
            name=name,
            tags=tags,
            solver_version=solver_version,
        )
        return new_surface_mesh

    def create_volume_mesh(
        self,
        name: str,
        params: VolumeMeshingParams,
        tags: List[str] = None,
        solver_version=None,
    ) -> VolumeMeshDraft:
        """
        Create volume mesh from surface mesh
        """

        return VolumeMeshDraft(
            name=name,
            surface_mesh_id=self.id,
            solver_version=solver_version,
            params=params,
            tags=tags,
        )


class SurfaceMeshList(Flow360ResourceListBase):
    """
    SurfaceMesh List component
    """

    def __init__(
        self,
        from_cloud: bool = True,
        include_deleted: bool = False,
        limit=100,
    ):
        super().__init__(
            ancestor_id=None,
            from_cloud=from_cloud,
            include_deleted=include_deleted,
            limit=limit,
            resourceClass=SurfaceMesh,
        )

    # pylint: disable=useless-parent-delegation
    def __getitem__(self, index) -> SurfaceMesh:
        """
        returns SurfaceMeshMeta item of the list
        """
        return super().__getitem__(index)

    # pylint: disable=useless-parent-delegation
    def __iter__(self) -> Iterator[SurfaceMesh]:
        return super().__iter__()
