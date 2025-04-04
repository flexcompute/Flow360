"""
Surface Mesh cloud asset (V2)
"""

from __future__ import annotations

import os
import threading
from enum import Enum
from typing import Any, List, Optional

import pydantic as pd

from flow360.cloud.flow360_requests import LengthUnitType, NewSurfaceMeshRequestV2
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import SurfaceMeshInterfaceV2
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.utils import (
    MeshNameParser,
    SurfaceMeshFile,
    shared_account_confirm_proceed,
)
from flow360.exceptions import Flow360FileError, Flow360ValueError
from flow360.log import log

from .simulation.primitives import Surface


class SurfaceMeshStatusV2(Enum):
    """Status of surface mesh resource, the is_final method is overloaded"""

    # pylint: disable=duplicate-code
    # We should unify the status enums for all resources and then remove the duplicate code warning.
    ERROR = "error"
    UPLOADED = "uploaded"
    UPLOADING = "uploading"
    RUNNING = "running"
    GENERATING = "generating"
    PROCESSED = "processed"
    DELETED = "deleted"
    PENDING = "pending"
    COMPLETED = "completed"

    def is_final(self):
        """
        Checks if status is final for geometry resource

        Returns
        -------
        bool
            True if status is final, False otherwise.
        """
        if self in [
            SurfaceMeshStatusV2.ERROR,
            SurfaceMeshStatusV2.PROCESSED,
            SurfaceMeshStatusV2.DELETED,
            SurfaceMeshStatusV2.COMPLETED,
        ]:
            return True
        return False


# pylint: disable=R0801
class SurfaceMeshMetaV2(AssetMetaBaseModelV2):
    """
    SurfaceMeshMeta component
    """

    file_name: Optional[str] = pd.Field(None, alias="fileName")
    status: SurfaceMeshStatusV2 = pd.Field()  # Overshadowing to ensure correct is_final() method


class SurfaceMeshDraftV2(ResourceDraft):
    """
    Surface mesh Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: str,
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        self._file_name = file_names
        self.project_name = project_name
        self.tags = tags if tags is not None else []
        self.length_unit = length_unit
        self.solver_version = solver_version
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_surface_mesh()

    def _validate_surface_mesh(self):
        if self._file_name is not None:
            try:
                SurfaceMeshFile(file_names=self._file_name)
            except pd.ValidationError as e:
                raise Flow360FileError(str(e)) from e

        if self.project_name is None:
            self.project_name = os.path.splitext(os.path.basename(self._file_name))[0]
            log.warning(
                "`project_name` is not provided. "
                f"Using the file name {self.project_name} as project name."
            )

        if self.length_unit not in LengthUnitType.__args__:
            raise Flow360ValueError(
                f"specified length_unit : {self.length_unit} is invalid. "
                f"Valid options are: {list(LengthUnitType.__args__)}"
            )

        if self.solver_version is None:
            raise Flow360ValueError("solver_version field is required.")

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(self, description="", progress_callback=None, run_async=False) -> SurfaceMeshV2:
        """
        Submit surface mesh file to cloud and create a new project

        Parameters
        ----------
        description : str, optional
            description of the project, by default ""
        progress_callback : callback, optional
            Use for custom progress bar, by default None
        run_async : bool, optional
            Whether to submit surface mesh asynchronously (default is False).

        Returns
        -------
        SurfaceMeshV2
            SurfaceMeshV2 object with id
        """

        self._validate()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        # The first geometry is assumed to be the main one.
        req = NewSurfaceMeshRequestV2(
            name=self.project_name,
            solver_version=self.solver_version,
            tags=self.tags,
            file_name=self._file_name,
            # pylint: disable=fixme
            # TODO: remove hardcoding
            parent_folder_id="ROOT.FLOW360",
            length_unit=self.length_unit,
            description=description,
        )

        ##:: Create new Geometry resource and project
        resp = RestApi(SurfaceMeshInterfaceV2.endpoint).post(req.dict())
        info = SurfaceMeshMetaV2(**resp)

        ##:: upload surface mesh file
        surface_mesh = SurfaceMeshV2(info.id)
        heartbeat_info = {"resourceId": info.id, "resourceType": "SurfaceMesh", "stop": False}
        # Keep posting the heartbeat to keep server patient about uploading.
        heartbeat_thread = threading.Thread(target=post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()

        surface_mesh._webapi._upload_file(
            remote_file_name=info.file_name,
            file_name=self._file_name,
            progress_callback=progress_callback,
        )

        mesh_parser = MeshNameParser(self._file_name)
        remote_mesh_parser = MeshNameParser(info.file_name)
        if mesh_parser.is_ugrid():
            # Upload the mapbc file too.
            expected_local_mapbc_file = mesh_parser.get_associated_mapbc_filename()
            if os.path.isfile(expected_local_mapbc_file):
                surface_mesh._webapi._upload_file(
                    remote_file_name=remote_mesh_parser.get_associated_mapbc_filename(),
                    file_name=mesh_parser.get_associated_mapbc_filename(),
                    progress_callback=progress_callback,
                )
            else:
                log.warning(
                    f"The expected mapbc file {expected_local_mapbc_file} specifying "
                    "user-specified boundary names doesn't exist."
                )

        heartbeat_info["stop"] = True
        heartbeat_thread.join()
        ##:: kick off pipeline
        surface_mesh._webapi._complete_upload()
        log.info(f"Surface mesh successfully submitted: {surface_mesh.short_description()}")
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        if run_async:
            return surface_mesh
        log.info("Waiting for surface mesh to be processed.")
        surface_mesh._webapi.get_info()
        # uses from_cloud to ensure all metadata is ready before yielding the object
        return SurfaceMeshV2.from_cloud(info.id)


class SurfaceMeshV2(AssetBase):
    """
    Surface mesh component for workbench (simulation V2)
    """

    _interface_class = SurfaceMeshInterfaceV2
    _meta_class = SurfaceMeshMetaV2
    _draft_class = SurfaceMeshDraftV2
    _web_api_class = Flow360Resource
    _entity_info_class = SurfaceMeshEntityInfo
    _cloud_resource_type_name = "SurfaceMesh"

    # pylint: disable=fixme
    # TODO: add _mesh_stats_file = "meshStats.json" like in VolumeMeshV2

    @classmethod
    # pylint: disable=redefined-builtin
    def from_cloud(cls, id: str, **kwargs) -> SurfaceMeshV2:
        """
        Parameters
        ----------
        id : str
            ID of the surface mesh resource in the cloud

        Returns
        -------
        SurfaceMeshV2
            Surface mesh object
        """
        asset_obj = super().from_cloud(id, **kwargs)

        return asset_obj

    @classmethod
    def from_local_storage(
        cls, mesh_id: str = None, local_storage_path="", meta_data: SurfaceMeshMetaV2 = None
    ) -> SurfaceMeshV2:
        """
        Parameters
        ----------
        mesh_id : str
            ID of the surface mesh resource

        local_storage_path:
            The folder of the project, defaults to current working directory

        Returns
        -------
        SurfaceMeshV2
            Surface mesh object
        """
        return super()._from_local_storage(
            asset_id=mesh_id, local_storage_path=local_storage_path, meta_data=meta_data
        )

    @classmethod
    # pylint: disable=too-many-arguments,arguments-renamed
    def from_file(
        cls,
        file_name: str,
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ) -> SurfaceMeshDraftV2:
        """
        Parameters
        ----------
        file_name : str
            The name of the input surface mesh file (*.cgns, *.ugrid)
        project_name : str, optional
            The name of the newly created project, defaults to file name if empty
        solver_version: str
            Solver version to use for the project
        length_unit: LengthUnitType
            Length unit to use for the project ("m", "mm", "cm", "inch", "ft")
        tags: List[str]
            List of string tags to be added to the project upon creation

        Returns
        -------
        SurfaceMeshDraftV2
            Draft of the surface mesh to be submitted
        """
        # For type hint only but proper fix is to fully abstract the Draft class too.
        return super().from_file(
            file_names=file_name,
            project_name=project_name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
        )

    # pylint: disable=useless-parent-delegation
    def get_default_settings(self, simulation_dict: dict):
        """Get the default surface mesh settings from the simulation dict"""
        return super().get_default_settings(simulation_dict)

    @property
    def boundary_names(self) -> List[str]:
        """
        Retrieve all boundary names available in this surface mesh as a list

        Returns
        -------
        List[str]
            List of boundary names contained within the surface mesh
        """
        self.internal_registry = self._entity_info.get_registry(self.internal_registry)

        return [
            surface.name for surface in self.internal_registry.get_bucket(by_type=Surface).entities
        ]

    def __getitem__(self, key: str):
        """
        Parameters
        ----------
        key : str
            The name of the entity to be found

        Returns
        -------
        Surface
            The boundary object
        """
        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        self.internal_registry = self._entity_info.get_registry(self.internal_registry)

        return self.internal_registry.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError("Assigning/setting entities is not supported.")
