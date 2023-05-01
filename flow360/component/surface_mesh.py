"""
Surface mesh component
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Iterator, List, Union

import pydantic as pd

from ..cloud.rest_api import RestApi
from ..cloud.s3_utils import S3TransferType
from ..exceptions import FileError as FlFileError
from ..exceptions import ValueError as FlValueError
from ..log import log
from .flow360_params.params_base import params_generic_validator
from .meshing.params import SurfaceMeshingParams, VolumeMeshingParams
from .resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    Flow360ResourceListBase,
    ResourceDraft,
)
from .utils import validate_type
from .validator import Validator
from .volume_mesh import VolumeMeshDraft


class SurfaceMeshDownloadable(Enum):
    """
    Surface Mesh downloadable files
    """

    CONFIG_JSON = "config.json"


# pylint: disable=E0213
class SurfaceMeshMeta(Flow360ResourceBaseModel, extra=pd.Extra.allow):
    """
    SurfaceMeshMeta component
    """

    params: Union[SurfaceMeshingParams, None, dict] = pd.Field(alias="config")

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


# pylint: disable=too-few-public-methods
class SurfaceMeshBase:
    """SurfaceMeshBase base class"""

    _endpoint = "surfacemeshes"


class SurfaceMeshDraft(SurfaceMeshBase, ResourceDraft):
    """
    Surface Mesh Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        geometry_file: str,
        params: SurfaceMeshingParams,
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
    ):
        self._geometry_file = geometry_file
        self.name = name
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self.params = params
        self._validate()
        self.params = params.copy(deep=True)
        ResourceDraft.__init__(self)

    def _validate(self):
        _, ext = os.path.splitext(self.geometry_file)
        if ext not in [".csm", ".egads"]:
            raise FlValueError(
                f"Unsupported geometry file extensions: {ext}. Supported: [csm, egads]."
            )

        if not os.path.exists(self.geometry_file):
            raise FlFileError(f"{self.geometry_file} not found.")

        if not isinstance(self.params, SurfaceMeshingParams):
            raise FlValueError(
                f"params must be of type: SurfaceMeshingParams, got {self.params}, type={type(self.params)} instead."
            )

    @property
    def geometry_file(self) -> str:
        """geometry file"""
        return self._geometry_file

    # pylint: disable=protected-access
    def submit(self, progress_callback=None) -> SurfaceMesh:
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
        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.geometry_file))[0]

        data = {
            "name": self.name,
            "tags": self.tags,
            "config": self.params.to_flow360_json(),
        }
        if self.solver_version:
            data["solverVersion"] = self.solver_version

        self.validator_api(self.params, solver_version=self.solver_version)

        resp = RestApi(self._endpoint).post(data)
        info = SurfaceMeshMeta(**resp)
        self._id = info.id
        submitted_mesh = SurfaceMesh(self.id)

        _, ext = os.path.splitext(self.geometry_file)
        remote_file_name = f"geometry{ext}"
        submitted_mesh.upload_file(
            remote_file_name, self.geometry_file, progress_callback=progress_callback
        )
        submitted_mesh._complete_upload(remote_file_name)
        log.info(f"SurfaceMesh successfully submitted: {submitted_mesh.short_description()}")
        return submitted_mesh

    @classmethod
    def validator_api(cls, params: SurfaceMeshingParams, solver_version=None):
        """
        validation api: validates surface meshing parameters before submitting
        """
        return Validator.SURFACE_MESH.validate(params, solver_version=solver_version)


class SurfaceMesh(SurfaceMeshBase, Flow360Resource):
    """
    Surface mesh component
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            resource_type="Surface Mesh",
            info_type_class=SurfaceMeshMeta,
            s3_transfer_method=S3TransferType.SURFACE_MESH,
            endpoint=self._endpoint,
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
    #     keep_folder: bool = True,
    #     overwrite: bool = True,
    #     progress_callback=None,
    # ):
    #     """
    #     Download file from surface mesh
    #     :param file_name:
    #     :param to_file:
    #     :param keep_folder:
    #     :return:
    #     """
    #     if isinstance(file_name, SurfaceMeshDownloadable):
    #         file_name = file_name.value
    #     return super().download_file(
    #         file_name,
    #         to_file,
    #         keep_folder=keep_folder,
    #         overwrite=overwrite,
    #         progress_callback=progress_callback,
    #     )

    # def download(self, to_file=".", keep_folder: bool = True):
    #     """
    #     Download surface mesh file
    #     :param to_file:
    #     :param keep_folder:
    #     :return:
    #     """
    #     super().download_file(, to_file, keep_folder)

    def download_log(self, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download_file("logs/flow360_surface_mesh.user.log", to_file, keep_folder)

    @classmethod
    def _meta_class(cls):
        """
        returns surface mesh meta info class: SurfaceMeshMeta
        """
        return SurfaceMeshMeta

    def _complete_upload(self, remote_file_name):
        """
        Complete surface mesh upload
        :return:
        """
        resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
        self._info = SurfaceMeshMeta(**resp)

    @classmethod
    def from_cloud(cls, surface_mesh_id: str):
        """
        Get surface mesh from cloud
        :param surface_mesh_id:
        :return:
        """
        return cls(surface_mesh_id)

    @classmethod
    def create(
        cls,
        geometry_file: str,
        params: SurfaceMeshingParams,
        name: str = None,
        tags: List[str] = None,
        solver_version: str = None,
    ) -> SurfaceMeshDraft:
        """ "Create new surface mesh from geometry"

        Parameters
        ----------
        geometry_file : str
            _description_
        params : SurfaceMeshingParams
            _description_
        name : str, optional
            _description_, by default None
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
            geometry_file=geometry_file,
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
