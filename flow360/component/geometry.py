"""
Geometry component
"""

from __future__ import annotations

import os
import threading
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic as pd

from flow360.cloud.flow360_requests import (
    GeometryFileMeta,
    LengthUnitType,
    NewGeometryDependencyRequest,
    NewGeometryRequest,
)
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import GeometryInterface
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
    SubmissionMode,
)
from flow360.component.simulation.folder import Folder
from flow360.component.simulation.primitives import Edge, GeometryBodyGroup, Surface
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    GeometryFiles,
    MeshNameParser,
    match_file_pattern,
    shared_account_confirm_proceed,
)
from flow360.exceptions import Flow360FileError, Flow360ValueError
from flow360.log import log

# Re-exports for face grouping API
from flow360.component.geometry_tree import TreeBackend, NodeSet, Node
from flow360.component.geometry_tree.face_group import FaceGroup
from flow360.component.import_geometry.import_geometry_api import ImportGeometryApi


def _is_file_path(value):
    """Detect if a string looks like a geometry file path rather than a UUID."""
    if not isinstance(value, str):
        return False
    if match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, value):
        return True
    if "/" in value or value.startswith("./") or value.startswith("../"):
        return True
    return False


class GeometryStatus(Enum):
    """Status of geometry resource, the is_final method is overloaded"""

    ERROR = "error"
    UPLOADED = "uploaded"
    UPLOADING = "uploading"
    RUNNING = "running"
    GENERATING = "generating"
    PROCESSED = "processed"
    DELETED = "deleted"
    PENDING = "pending"
    UNKNOWN = "unknown"

    def is_final(self):
        """
        Checks if status is final for geometry resource

        Returns
        -------
        bool
            True if status is final, False otherwise.
        """
        if self in [
            GeometryStatus.ERROR,
            GeometryStatus.PROCESSED,
            GeometryStatus.DELETED,
        ]:
            return True
        return False


# pylint: disable=R0801
class GeometryMeta(AssetMetaBaseModelV2):
    """
    GeometryMeta component
    """

    status: GeometryStatus = pd.Field()  # Overshadowing to ensure correct is_final() method
    dependency: bool = pd.Field(False)


class GeometryDraft(ResourceDraft):
    """
    Unified Geometry Draft component for uploading geometry files.

    This class handles both:
    - Creating a new project with geometry as the root asset
    - Adding geometry as a dependency to an existing project

    The submission mode is determined by how the draft is created (via factory methods
    on the Geometry class) and affects the behavior of the submit() method.

    All geometries are conceptually equivalent - they are components that can be used
    to create the final geometry for simulation. The distinction between "root" and
    "dependency" is only about where the geometry is uploaded (new project vs existing
    project), not about any fundamental difference in the geometry itself.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        file_names: Union[List[str], str],
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        folder: Optional[Folder] = None,
    ):
        """
        Initialize a GeometryDraft with common attributes.

        For creating a new project (root geometry):
            Use Geometry.from_file() which sets project_name, solver_version, folder

        For adding to existing project (dependency geometry):
            Use Geometry.import_to_project() which sets the dependency context

        Parameters
        ----------
        file_names : Union[List[str], str]
            Path(s) to the geometry file(s)
        length_unit : LengthUnitType, optional
            Unit of length (default is "m")
        tags : List[str], optional
            Tags to assign to the geometry (default is None)
        project_name : str, optional
            Name of the project (for project root mode)
        solver_version : str, optional
            Solver version (for project root mode)
        folder : Optional[Folder], optional
            Parent folder (for project root mode)
        """
        self._file_names = file_names
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

        self._validate_geometry()
        ResourceDraft.__init__(self)

    def _validate_geometry(self):
        """Validate geometry files and length unit."""
        if not isinstance(self.file_names, list) or len(self.file_names) == 0:
            raise Flow360FileError("file_names field has to be a non-empty list.")

        try:
            GeometryFiles(file_names=self.file_names)
        except pd.ValidationError as e:
            raise Flow360FileError(str(e)) from e

        for geometry_file in self.file_names:
            if not os.path.exists(geometry_file):
                raise Flow360FileError(f"{geometry_file} not found.")

        if self.length_unit not in LengthUnitType.__args__:
            raise Flow360ValueError(
                f"specified length_unit : {self.length_unit} is invalid. "
                f"Valid options are: {list(LengthUnitType.__args__)}"
            )

    def _set_default_project_name(self):
        """Set default project name if not provided for project creation."""
        if self.project_name is None:
            self.project_name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
            log.warning(
                "`project_name` is not provided. "
                f"Using the first geometry file name {self.project_name} as project name."
            )

    def _validate_submission_context(self):
        """Validate context for submission based on mode."""
        if self._submission_mode is None:
            raise ValueError("[Internal] Geometry submission context not set.")
        if self._submission_mode == SubmissionMode.PROJECT_ROOT and self.solver_version is None:
            raise Flow360ValueError("solver_version field is required.")
        if self._submission_mode == SubmissionMode.PROJECT_DEPENDENCY:
            if self.dependency_name is None or self.dependency_project_id is None:
                raise ValueError(
                    "[Internal] Dependency name and project ID must be set for geometry dependency submission."
                )

    @property
    def file_names(self) -> List[str]:
        """Geometry file paths as a list."""
        if isinstance(self._file_names, str):
            return [self._file_names]
        return self._file_names

    def set_dependency_context(
        self,
        name: str,
        project_id: str,
    ) -> None:
        """
        Configure this draft to add geometry to an existing project.

        Called internally by Geometry.import_to_project().
        """
        self._submission_mode = SubmissionMode.PROJECT_DEPENDENCY
        self.dependency_name = name
        self.dependency_project_id = project_id

    def _preprocess_mapbc_files(self) -> List[str]:
        """Find and return associated mapbc files for UGRID geometry files."""
        mapbc_files = []
        for file_path in self.file_names:
            mesh_parser = MeshNameParser(file_path)
            if mesh_parser.is_ugrid() and os.path.isfile(
                mesh_parser.get_associated_mapbc_filename()
            ):
                file_name_mapbc = mesh_parser.get_associated_mapbc_filename()
                mapbc_files.append(file_name_mapbc)
        return mapbc_files

    def _create_project_root_resource(
        self, mapbc_files: List[str], description: str = ""
    ) -> GeometryMeta:
        """Create a new geometry resource that will be the root of a new project."""
        self._set_default_project_name()
        req = NewGeometryRequest(
            name=self.project_name,
            solver_version=self.solver_version,
            tags=self.tags,
            files=[
                GeometryFileMeta(
                    name=os.path.basename(file_path),
                    type="main",
                )
                for file_path in self.file_names + mapbc_files
            ],
            parent_folder_id=self.folder.id if self.folder else "ROOT.FLOW360",
            length_unit=self.length_unit,
            description=description,
        )

        resp = RestApi(GeometryInterface.endpoint).post(req.dict())
        return GeometryMeta(**resp)

    def _create_dependency_resource(
        self, mapbc_files: List[str], description: str = "", draft_id: str = "", icon: str = ""
    ) -> GeometryMeta:
        """Create a geometry resource as a dependency in an existing project."""

        req = NewGeometryDependencyRequest(
            name=self.dependency_name,
            project_id=self.dependency_project_id,
            draft_id=draft_id,
            files=[
                GeometryFileMeta(
                    name=os.path.basename(file_path),
                    type="main",
                )
                for file_path in self.file_names + mapbc_files
            ],
            length_unit=self.length_unit,
            tags=self.tags,
            description=description,
            icon=icon,
        )

        resp = RestApi(GeometryInterface.endpoint).post(req.dict(), method="dependency")
        return GeometryMeta(**resp)

    def _upload_files(
        self,
        info: GeometryMeta,
        mapbc_files: List[str],
        progress_callback=None,
    ) -> Geometry:
        """Upload geometry files to the cloud."""
        # pylint: disable=protected-access
        geometry = Geometry(info.id)
        heartbeat_info = {"resourceId": info.id, "resourceType": "Geometry", "stop": False}

        # Keep posting the heartbeat to keep server patient about uploading.
        heartbeat_thread = threading.Thread(target=post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()

        try:
            for file_path in self.file_names + mapbc_files:
                geometry._webapi._upload_file(
                    remote_file_name=os.path.basename(file_path),
                    file_name=file_path,
                    progress_callback=progress_callback,
                )
        finally:
            heartbeat_info["stop"] = True
            heartbeat_thread.join()

        # Kick off pipeline
        geometry._webapi._complete_upload()

        # Setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id

        return geometry

    # pylint: disable=duplicate-code
    def submit(
        self,
        description: str = "",
        progress_callback=None,
        run_async: bool = False,
        draft_id: str = "",
        icon: str = "",
    ) -> Geometry:
        """
        Submit geometry to cloud.

        The behavior depends on how this draft was created:
        - If created via Geometry.from_file(): Creates a new project with this geometry as root
        - If created via Geometry.import_to_project(): Adds geometry to an existing project

        Parameters
        ----------
        description : str, optional
            Description of the geometry/project (default is "")
        progress_callback : callback, optional
            Use for custom progress bar (default is None)
        run_async : bool, optional
            Whether to return immediately after upload without waiting for processing
            (default is False)
        draft_id : str, optional
            ID of the draft to add geometry to (only used for dependency mode, default is "")
        icon : str, optional
            Icon for the geometry (only used for dependency mode, default is "")

        Returns
        -------
        Geometry
            Geometry object with id

        Raises
        ------
        Flow360ValueError
            If submission context is not set or user aborts
        """

        self._validate_geometry()
        self._validate_submission_context()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        mapbc_files = self._preprocess_mapbc_files()

        # Create the geometry resource based on submission mode
        if self._submission_mode == SubmissionMode.PROJECT_ROOT:
            info = self._create_project_root_resource(mapbc_files, description)
            log_message = "Geometry successfully submitted"
        else:
            info = self._create_dependency_resource(mapbc_files, description, draft_id, icon)
            log_message = "New geometry successfully submitted to the project"

        # Upload files
        geometry = self._upload_files(info, mapbc_files, progress_callback)

        log.info(f"{log_message}: {geometry.short_description()}")

        if run_async:
            return geometry

        log.info("Waiting for geometry to be processed.")
        return Geometry.from_cloud(info.id)


class Geometry(AssetBase):
    """
    Geometry component for workbench (simulation V2)
    """

    _interface_class = GeometryInterface
    _meta_class = GeometryMeta
    _draft_class = GeometryDraft
    _web_api_class = Flow360Resource
    _cloud_resource_type_name = "Geometry"

    # pylint: disable=redefined-builtin
    def __init__(self, id: Union[str, None], name: str = None):
        self._backend = None  # TreeBackend for face grouping
        self._tree_groups = {}  # name -> FaceGroup
        if id is not None and _is_file_path(id):
            self._init_from_file(id, name=name)
        else:
            super().__init__(id)
            self.snappy_body_registry = None

    @staticmethod
    def _make_timestamped_name(file_path):
        """Generate a resource name like 'MM-DD-hh-mm-ss-<filename>'."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return f"{timestamp}-{basename}"

    def _init_from_file(self, file_path, name=None):
        """Initialize from a geometry file via the import-geometry workflow."""
        api = ImportGeometryApi()

        if name is None:
            name = self._make_timestamped_name(file_path)

        # Step 1: Create import-geometry resource
        log.info(f"Creating import-geometry resource '{name}' for {file_path}")
        result = api.create(file_path, name=name)
        geometry_id = result["geometryId"]
        upload_urls = result["uploadUrls"]

        # Step 2: Upload file to presigned URL
        log.info(f"Uploading {file_path}...")
        api.upload_file(file_path, upload_urls[0])

        # Step 3: Mark upload complete
        api.complete_upload(geometry_id)

        # Step 4: Wait until processed
        log.info("Waiting for geometry tree processing...")
        api.wait_until_processed(geometry_id)

        # Step 5: Fetch tree
        tree_data = api.fetch_tree(geometry_id)

        # Step 6: Initialize base class with real geometry ID
        super(Geometry, self).__init__(geometry_id)
        self.snappy_body_registry = None

        # Step 7: Load tree into backend
        self._backend = TreeBackend()
        self._backend.load_from_json(tree_data)
        log.info(f"Geometry loaded: {len(self.faces())} faces")

    # ================================================================
    # Tree Navigation Methods
    # ================================================================

    def root_node(self) -> NodeSet:
        """Get NodeSet containing the root node."""
        if self._backend is None:
            raise Flow360ValueError(
                "Geometry tree not loaded. Use Geometry(file_path) to load from file."
            )
        root_id = self._backend.get_root()
        if root_id is None:
            return NodeSet(self, self._backend, set())
        return NodeSet(self, self._backend, {root_id})

    def children(self, **filters) -> NodeSet:
        """Get direct children of the root node."""
        return self.root_node().children(**filters)

    def descendants(self, **filters) -> NodeSet:
        """Get all descendants of the root."""
        return self.root_node().descendants(**filters)

    def faces(self, **filters) -> NodeSet:
        """Get all face nodes in the geometry."""
        return self.root_node().faces(**filters)

    # ================================================================
    # Face Group Management
    # ================================================================

    def create_face_group(self, name: str, selection: NodeSet) -> FaceGroup:
        """
        Create a named face group from a selection.

        Each face can only belong to one group. Faces in the selection
        are removed from any previous group they belonged to.
        """
        if name in self._tree_groups:
            raise ValueError(f"Group '{name}' already exists")

        # Extract face node IDs from the selection
        face_nodes = selection.faces()
        face_node_ids = face_nodes._node_ids

        # Remove these faces from any existing groups (exclusive ownership)
        for group in self._tree_groups.values():
            group._node_ids -= face_node_ids

        group = FaceGroup(name, face_node_ids)
        self._tree_groups[name] = group
        return group

    def get_face_group(self, name: str) -> FaceGroup:
        """Get a face group by name."""
        if name not in self._tree_groups:
            raise KeyError(f"Group '{name}' not found")
        return self._tree_groups[name]

    def list_groups(self):
        """List all group names."""
        return list(self._tree_groups.keys())

    def clear_groups(self) -> None:
        """Remove all face groups."""
        self._tree_groups.clear()

    # ================================================================
    # Set Operations
    # ================================================================

    def __sub__(self, other) -> NodeSet:
        """Subtract faces from total geometry (geometry - FaceGroup or NodeSet)."""
        all_faces = self.faces()
        if isinstance(other, FaceGroup):
            other_nodes = NodeSet(self, self._backend, other._node_ids)
            return all_faces - other_nodes
        elif isinstance(other, NodeSet):
            return all_faces - other.faces()
        else:
            return NotImplemented

    def __repr__(self) -> str:
        if self._backend is not None:
            return f"Geometry({len(self.faces())} faces)"
        return f"Geometry('{self.id}')"

    @property
    def face_group_tag(self):
        "getter for face_group_tag"
        return self._entity_info.face_group_tag

    @face_group_tag.setter
    def face_group_tag(self, new_value: str):
        raise SyntaxError("Cannot set face_group_tag, use group_faces_by_tag() instead.")

    @property
    def edge_group_tag(self):
        "getter for edge_group_tag"
        return self._entity_info.edge_group_tag

    @edge_group_tag.setter
    def edge_group_tag(self, new_value: str):
        raise SyntaxError("Cannot set edge_group_tag, use group_edges_by_tag() instead.")

    @property
    def body_group_tag(self):
        "getter for body_group_tag"
        return self._entity_info.body_group_tag

    @body_group_tag.setter
    def body_group_tag(self, new_value: str):
        raise SyntaxError("Cannot set body_group_tag, use group_bodies_by_tag() instead.")

    @property
    def snappy_bodies(self):
        """Getter for the snappy registry."""
        if self.snappy_body_registry is None:
            raise Flow360ValueError(
                "The faces in geometry are not grouped for snappy."
                "Please use `group_faces_for_snappy` function to group them first."
            )
        return self.snappy_body_registry

    def get_dynamic_default_settings(self, simulation_dict: dict):
        """Get the default geometry settings from the simulation dict"""

        def _get_default_geometry_accuracy(simulation_dict: dict) -> LengthType.Positive:
            """Get the default geometry accuracy from the simulation json"""
            if simulation_dict.get("meshing") is None:
                return None
            if simulation_dict["meshing"].get("defaults") is None:
                return None
            if simulation_dict["meshing"]["defaults"].get("geometry_accuracy") is None:
                return None
            # pylint: disable=no-member
            return LengthType.validate(simulation_dict["meshing"]["defaults"]["geometry_accuracy"])

        self.default_settings["geometry_accuracy"] = (
            self._entity_info.default_geometry_accuracy
            if self._entity_info.default_geometry_accuracy
            else _get_default_geometry_accuracy(simulation_dict=simulation_dict)
        )

    @classmethod
    # pylint: disable=redefined-builtin
    def from_cloud(cls, id: str, **kwargs) -> Geometry:
        """Create asset with the given ID"""
        asset_obj = super().from_cloud(id, **kwargs)
        return asset_obj

    @classmethod
    # pylint: disable=too-many-arguments
    def from_file(
        cls,
        file_names: Union[List[str], str],
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        folder: Optional[Folder] = None,
    ) -> GeometryDraft:
        # For type hint only but proper fix is to fully abstract the Draft class too.
        return super().from_file(
            file_names, project_name, solver_version, length_unit, tags, folder=folder
        )

    @classmethod
    # pylint: disable=too-many-arguments
    def import_to_project(
        cls,
        name: str,
        file_names: Union[List[str], str],
        project_id: str,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ) -> GeometryDraft:
        """
        Create a geometry draft for adding to an existing project.

        This creates a geometry that will be added as a supplementary component
        (dependency) to an existing project, rather than creating a new project.

        Parameters
        ----------
        name : str
            Name for the geometry component
        file_names : Union[List[str], str]
            Path(s) to the geometry file(s)
        project_id : str
            ID of the existing project to add this geometry to
        length_unit : LengthUnitType, optional
            Unit of length (default is "m")
        tags : List[str], optional
            Tags to assign to the geometry (default is None)

        Returns
        -------
        GeometryDraft
            A draft configured for submission to an existing project
        """
        draft = GeometryDraft(
            file_names=file_names,
            length_unit=length_unit,
            tags=tags,
        )
        draft.set_dependency_context(name=name, project_id=project_id)
        return draft

    def show_available_groupings(self, verbose_mode: bool = False):
        """Display all the possible groupings for faces and edges"""
        self._show_available_entity_groups(
            "faces",
            ignored_attribute_tags=["__all__", "faceId"],
            show_ids_in_each_group=verbose_mode,
        )
        self._show_available_entity_groups(
            "edges",
            ignored_attribute_tags=["__all__", "edgeId"],
            show_ids_in_each_group=verbose_mode,
        )
        self._show_available_entity_groups(
            "bodies",
            ignored_attribute_tags=["__all__", "bodyId"],
            show_ids_in_each_group=verbose_mode,
        )

    @classmethod
    def from_local_storage(
        cls, geometry_id: str = None, local_storage_path="", meta_data: GeometryMeta = None
    ) -> Geometry:
        """
        Parameters
        ----------
        geometry_id : str
            ID of the geometry resource

        local_storage_path:
            The folder of the project, defaults to current working directory

        Returns
        -------
        Geometry
            Geometry object
        """

        return super()._from_local_storage(
            asset_id=geometry_id, local_storage_path=local_storage_path, meta_data=meta_data
        )

    def _show_available_entity_groups(
        self,
        entity_type_name: Literal["faces", "edges", "bodies"],
        ignored_attribute_tags: list = None,
        show_ids_in_each_group: bool = False,
    ) -> None:
        """
        Display all the grouping info for the given entity type
        """

        if entity_type_name not in ["faces", "edges", "bodies"]:
            raise Flow360ValueError(
                f"entity_type_name: {entity_type_name} is invalid. Valid options are: ['faces', 'edges', 'bodies']"
            )

        # pylint: disable=no-member
        if entity_type_name == "faces":
            attribute_names = self._entity_info.face_attribute_names
            grouped_items = self._entity_info.grouped_faces
        elif entity_type_name == "edges":
            attribute_names = self._entity_info.edge_attribute_names
            grouped_items = self._entity_info.grouped_edges
        else:
            attribute_names = self._entity_info.body_attribute_names
            grouped_items = self._entity_info.grouped_bodies

        log.info(f" >> Available attribute tags for grouping **{entity_type_name}**:")
        for tag_index, attribute_tag in enumerate(attribute_names):
            if ignored_attribute_tags is not None and attribute_tag in ignored_attribute_tags:
                continue
            log.info(
                f"    >> Tag '{tag_index}': {attribute_tag}. Grouping with this tag results in:"
            )
            for index, entity in enumerate(grouped_items[tag_index]):
                log.info(f"        >> [{index}]: {entity.name}")
                if show_ids_in_each_group is True:
                    log.info(f"           IDs: {entity.private_attribute_sub_components}")

    def group_faces_by_tag(self, tag_name: str) -> None:
        """
        Group faces by tag name
        """
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._group_entity_by_tag(
            "face", tag_name, self.internal_registry
        )

    def group_edges_by_tag(self, tag_name: str) -> None:
        """
        Group edges by tag name
        """
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._group_entity_by_tag(
            "edge", tag_name, self.internal_registry
        )

    def group_bodies_by_tag(self, tag_name: str) -> None:
        """
        Group bodies by tag name
        """
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._group_entity_by_tag(
            "body", tag_name, self.internal_registry
        )

    def group_faces_for_snappy(self) -> None:
        """
        Group faces according to body::region convention for snappyHexMesh.
        """
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._group_entity_by_tag(
            "face", "faceId", self.internal_registry
        )
        # pylint: disable=protected-access
        self.snappy_body_registry = self._entity_info._group_faces_by_snappy_format()

    def reset_face_grouping(self) -> None:
        """Reset the face grouping"""
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._reset_grouping("face", self.internal_registry)
        if self.snappy_body_registry is not None:
            self.snappy_body_registry = self.snappy_body._reset_grouping(
                "face", self.snappy_body_registry
            )

    def reset_edge_grouping(self) -> None:
        """Reset the edge grouping"""
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._reset_grouping("edge", self.internal_registry)

    def reset_body_grouping(self) -> None:
        """Reset the body grouping"""
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._reset_grouping("body", self.internal_registry)

    def _rename_entity(
        self,
        entity_type_name: Literal["face", "edge", "body"],
        current_name_pattern: str,
        new_name_prefix: str,
    ):
        """
        Rename the entity

        Parameters
        ----------
        entity_type_name : Literal["face", "edge", "body"]
            The type of entity that needs renaming

        current_name_pattern:
            The current name of a single entity or the name pattern of the entities

        new_name_prefix:
            The new name of a single entity or the new name prefix of the entities

        """

        # pylint: disable=too-many-boolean-expressions
        if (
            (entity_type_name == "face" and not self.face_group_tag)
            or (entity_type_name == "edge" and not self.edge_group_tag)
            or (entity_type_name == "body" and not self.body_group_tag)
        ):
            raise Flow360ValueError(
                f"Renaming failed: Could not find {entity_type_name} grouping info in the draft's simulation settings."
                "Please group them first before renaming the entities."
            )

        matched_entities = self.internal_registry.find_by_naming_pattern(
            pattern=current_name_pattern
        )
        if entity_type_name == "body":
            matched_entities = [
                entity for entity in matched_entities if isinstance(entity, GeometryBodyGroup)
            ]
        if entity_type_name == "face":
            matched_entities = [
                entity for entity in matched_entities if isinstance(entity, Surface)
            ]
        if entity_type_name == "edge":
            matched_entities = [entity for entity in matched_entities if isinstance(entity, Edge)]

        matched_entities = sorted(
            matched_entities,
            key=lambda x: x.name,
        )
        if len(matched_entities) == 0:
            raise Flow360ValueError(
                f"Renaming failed: No entity is found to match the input name pattern: {current_name_pattern}."
            )

        for idx, entity in enumerate(matched_entities):
            new_name = (
                f"{new_name_prefix}_{(idx+1):04d}" if len(matched_entities) > 1 else new_name_prefix
            )
            if self.internal_registry.find_by_naming_pattern(new_name):
                raise Flow360ValueError(
                    f"Renaming failed: An entity with the new name: {new_name} already exists."
                )
            with model_attribute_unlock(entity, "name"):
                entity.name = new_name

    def rename_edges(self, current_name_pattern: str, new_name_prefix: str):
        """
        Rename the edge in the current edge group

        Parameters
        ----------
        current_name_pattern:
            The current name of a single edge or the name pattern of the edges

        new_name_prefix:
            The new name of a single edge or the new name prefix of the edges
        """
        self._rename_entity(
            entity_type_name="edge",
            current_name_pattern=current_name_pattern,
            new_name_prefix=new_name_prefix,
        )

    def rename_surfaces(self, current_name_pattern: str, new_name_prefix: str):
        """
        Rename the face in the current face group

        Parameters
        ----------
        current_name_pattern:
            The current name of a single face or the name pattern of the faces

        new_name_prefix:
            The new name of a single face or the new name prefix of the faces
        """
        self._rename_entity(
            entity_type_name="face",
            current_name_pattern=current_name_pattern,
            new_name_prefix=new_name_prefix,
        )

    def rename_body_groups(self, current_name_pattern: str, new_name_prefix: str):
        """
        Rename the body in the current body group

        Parameters
        ----------
        current_name_pattern:
            The current name of a single body or the name pattern of the bodies

        new_name_prefix:
            The new name of a single body or the new name prefix of the bodies
        """
        self._rename_entity(
            entity_type_name="body",
            current_name_pattern=current_name_pattern,
            new_name_prefix=new_name_prefix,
        )

    def __getitem__(self, key: str):
        """
        Get the entity by name.
        `key` is the name of the entity or the naming pattern if wildcard is used.
        """
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.draft_context import get_active_draft

        if get_active_draft() is not None:
            log.warning(
                "Accessing entities via asset[key] while a DraftContext is active. "
                "Use draft.surfaces[key] or draft.body_groups[key] instead to ensure "
                "modifications are tracked in the draft's entity_info."
            )

        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        if hasattr(self, "internal_registry") is False or self.internal_registry is None:
            raise Flow360ValueError(
                "The faces/edges/bodies in geometry are not grouped yet."
                "Please use `group_faces_by_tag` or `group_edges_by_tag` function to group them first."
            )
            # Note: Or we assume group default by just FaceID and EdgeID? Not sure if this is actually useful.
        return self.internal_registry.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError("Assigning/setting entities is not supported.")
