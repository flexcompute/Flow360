"""
Surface Mesh cloud asset (V2)
"""

from __future__ import annotations

import os
import threading
from enum import Enum
from typing import Any, List, Optional

import pydantic as pd

from flow360.cloud.flow360_requests import (
    LengthUnitType,
    NewSurfaceMeshDependencyRequest,
    NewSurfaceMeshRequestV2,
)
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import SurfaceMeshInterfaceV2
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
    SubmissionMode,
)
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.folder import Folder
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
    dependency: bool = pd.Field(False)


class SurfaceMeshDraftV2(ResourceDraft):
    """
    Unified Surface Mesh Draft component for uploading surface mesh files.

    This class handles both:
    - Creating a new project with surface mesh as the root asset
    - Adding surface mesh as a dependency to an existing project

    The submission mode is determined by how the draft is created (via factory methods
    on the SurfaceMeshV2 class) and affects the behavior of the submit() method.

    All surface meshes are conceptually equivalent - they are components that can be used
    to create the final mesh for simulation. The distinction between "root" and
    "dependency" is only about where the surface mesh is uploaded (new project vs existing
    project), not about any fundamental difference in the surface mesh itself.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        file_names: str,
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        folder: Optional[Folder] = None,
    ):
        """
        Initialize a SurfaceMeshDraftV2 with common attributes.

        For creating a new project (root surface mesh):
            Use SurfaceMeshV2.from_file() which sets project_name, solver_version, folder

        For adding to existing project (dependency surface mesh):
            Use SurfaceMeshV2.import_to_project() which sets the dependency context

        Parameters
        ----------
        file_names : str
            Path to the surface mesh file
        length_unit : LengthUnitType, optional
            Unit of length (default is "m")
        tags : List[str], optional
            Tags to assign to the surface mesh (default is None)
        project_name : str, optional
            Name of the project (for project root mode)
        solver_version : str, optional
            Solver version (for project root mode)
        folder : Optional[Folder], optional
            Parent folder (for project root mode)
        """
        self._file_name = file_names
        self.project_name = project_name
        self.tags = tags if tags is not None else []
        self.length_unit = length_unit
        self.solver_version = solver_version
        self.folder = folder

        # pylint: disable=fixme
        # TODO: create a DependableResourceDraft for GeometryDraft and SurfaceMeshDraft
        self.dependency_name = None
        self.dependency_project_id = None
        self._submission_mode: SubmissionMode = SubmissionMode.PROJECT_ROOT

        self._validate_surface_mesh()
        ResourceDraft.__init__(self)

    def _validate_surface_mesh(self):
        """Validate surface mesh file and length unit."""
        if self._file_name is not None:
            try:
                SurfaceMeshFile(file_names=self._file_name)
            except pd.ValidationError as e:
                raise Flow360FileError(str(e)) from e

        if self.length_unit not in LengthUnitType.__args__:
            raise Flow360ValueError(
                f"specified length_unit : {self.length_unit} is invalid. "
                f"Valid options are: {list(LengthUnitType.__args__)}"
            )

    def _set_default_project_name(self):
        """Set default project name if not provided for project creation."""
        if self.project_name is None:
            self.project_name = os.path.splitext(os.path.basename(self._file_name))[0]
            log.warning(
                "`project_name` is not provided. "
                f"Using the file name {self.project_name} as project name."
            )

    def _validate_submission_context(self):
        """Validate context for submission based on mode."""
        if self._submission_mode is None:
            raise ValueError("[Internal] Surface Mesh Submission context not set.")
        if self._submission_mode == SubmissionMode.PROJECT_ROOT and self.solver_version is None:
            raise Flow360ValueError("solver_version field is required.")
        if self._submission_mode == SubmissionMode.PROJECT_DEPENDENCY:
            if self.dependency_name is None or self.dependency_project_id is None:
                raise ValueError(
                    "[Internal] Dependency name and project ID must be set for surface mesh dependency submission."
                )

    def set_dependency_context(
        self,
        name: str,
        project_id: str,
    ) -> None:
        """
        Configure this draft to add surface mesh to an existing project.

        Called internally by SurfaceMeshV2.import_to_project().
        """
        self._submission_mode = SubmissionMode.PROJECT_DEPENDENCY
        self.dependency_name = name
        self.dependency_project_id = project_id

    def _create_project_root_resource(self, description: str = "") -> SurfaceMeshMetaV2:
        """Create a new surface mesh resource that will be the root of a new project."""

        self._set_default_project_name()
        req = NewSurfaceMeshRequestV2(
            name=self.project_name,
            solver_version=self.solver_version,
            tags=self.tags,
            file_name=self._file_name,
            parent_folder_id=self.folder.id if self.folder else "ROOT.FLOW360",
            length_unit=self.length_unit,
            description=description,
        )

        resp = RestApi(SurfaceMeshInterfaceV2.endpoint).post(req.dict())
        return SurfaceMeshMetaV2(**resp)

    def _create_dependency_resource(
        self, description: str = "", draft_id: str = "", icon: str = ""
    ) -> SurfaceMeshMetaV2:
        """Create a surface mesh resource as a dependency in an existing project."""

        req = NewSurfaceMeshDependencyRequest(
            name=self.dependency_name,
            project_id=self.dependency_project_id,
            draft_id=draft_id,
            file_name=self._file_name,
            length_unit=self.length_unit,
            tags=self.tags,
            description=description,
            icon=icon,
        )

        resp = RestApi(SurfaceMeshInterfaceV2.endpoint).post(req.dict(), method="dependency")
        return SurfaceMeshMetaV2(**resp)

    def _upload_files(
        self,
        info: SurfaceMeshMetaV2,
        progress_callback=None,
    ) -> SurfaceMeshV2:
        """Upload surface mesh files to the cloud."""
        # pylint: disable=protected-access
        surface_mesh = SurfaceMeshV2(info.id)
        heartbeat_info = {"resourceId": info.id, "resourceType": "SurfaceMesh", "stop": False}

        # Keep posting the heartbeat to keep server patient about uploading.
        heartbeat_thread = threading.Thread(target=post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()

        try:
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
        finally:
            heartbeat_info["stop"] = True
            heartbeat_thread.join()

        # Kick off pipeline
        surface_mesh._webapi._complete_upload()

        # Setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id

        return surface_mesh

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(
        self,
        description: str = "",
        progress_callback=None,
        run_async: bool = False,
        draft_id: str = "",
        icon: str = "",
    ) -> SurfaceMeshV2:
        """
        Submit surface mesh to cloud.

        The behavior depends on how this draft was created:
        - If created via SurfaceMeshV2.from_file(): Creates a new project with this surface mesh as root
        - If created via SurfaceMeshV2.import_to_project(): Adds surface mesh to an existing project

        Parameters
        ----------
        description : str, optional
            Description of the surface mesh/project (default is "")
        progress_callback : callback, optional
            Use for custom progress bar (default is None)
        run_async : bool, optional
            Whether to return immediately after upload without waiting for processing
            (default is False)
        draft_id : str, optional
            ID of the draft to add surface mesh to (only used for dependency mode, default is "")
        icon : str, optional
            Icon for the surface mesh (only used for dependency mode, default is "")

        Returns
        -------
        SurfaceMeshV2
            SurfaceMeshV2 object with id

        Raises
        ------
        Flow360ValueError
            If submission context is not set or user aborts
        """

        self._validate_surface_mesh()
        self._validate_submission_context()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        # Create the surface mesh resource based on submission mode
        if self._submission_mode == SubmissionMode.PROJECT_ROOT:
            info = self._create_project_root_resource(description)
            log_message = "Surface mesh successfully submitted"
        else:
            info = self._create_dependency_resource(description, draft_id, icon)
            log_message = "New surface mesh successfully submitted to the project"

        # Upload files
        surface_mesh = self._upload_files(info, progress_callback)

        log.info(f"{log_message}: {surface_mesh.short_description()}")

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
        folder: Optional[Folder] = None,
    ) -> SurfaceMeshDraftV2:
        """
        Parameters
        ----------
        file_name : str
            The name of the input surface mesh file (``*.cgns``, ``*.ugrid``)
        project_name : str, optional
            The name of the newly created project, defaults to file name if empty
        solver_version: str
            Solver version to use for the project
        length_unit: LengthUnitType
            Length unit to use for the project ("m", "mm", "cm", "inch", "ft")
        tags: List[str]
            List of string tags to be added to the project upon creation
        folder : Optional[Folder], optional
            Parent folder for the project. If None, creates in root.

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
            folder=folder,
        )

    @classmethod
    # pylint: disable=too-many-arguments
    def import_to_project(
        cls,
        name: str,
        file_name: str,
        project_id: str,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ) -> SurfaceMeshDraftV2:
        """
        Create a surface mesh draft for adding to an existing project.

        This creates a surface mesh that will be added as a supplementary component
        (dependency) to an existing project, rather than creating a new project.

        Parameters
        ----------
        name : str
            Name for the surface mesh component
        file_name : str
            Path to the surface mesh file (*.cgns, *.ugrid)
        project_id : str
            ID of the existing project to add this surface mesh to
        length_unit : LengthUnitType, optional
            Unit of length (default is "m")
        tags : List[str], optional
            Tags to assign to the surface mesh (default is None)

        Returns
        -------
        SurfaceMeshDraftV2
            A draft configured for submission to an existing project
        """
        draft = SurfaceMeshDraftV2(
            file_names=file_name,
            length_unit=length_unit,
            tags=tags,
        )
        draft.set_dependency_context(name=name, project_id=project_id)
        return draft

    # pylint: disable=useless-parent-delegation
    def get_dynamic_default_settings(self, simulation_dict: dict):
        """Get the default surface mesh settings from the simulation dict"""
        return super().get_dynamic_default_settings(simulation_dict)

    @property
    def boundary_names(self) -> List[str]:
        """
        Retrieve all boundary names available in this surface mesh as a list

        Returns
        -------
        List[str]
            List of boundary names contained within the surface mesh
        """
        self.internal_registry = self._entity_info.get_persistent_entity_registry(
            self.internal_registry
        )
        # pylint: disable=protected-access
        return [surface.name for surface in self.internal_registry.view(Surface)._entities]

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
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.draft_context import get_active_draft

        if get_active_draft() is not None:
            log.warning(
                "Accessing entities via asset[key] while a DraftContext is active. "
                "Use draft.surfaces[key] instead to ensure "
                "modifications are tracked in the draft's entity_info."
            )

        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        self.internal_registry = self._entity_info.get_persistent_entity_registry(
            self.internal_registry
        )

        return self.internal_registry.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError("Assigning/setting entities is not supported.")
