"""
Geometry component
"""

from __future__ import annotations

import os
import threading
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic as pd

from flow360.cloud.flow360_requests import (
    GeometryFileMeta,
    LengthUnitType,
    NewGeometryRequest,
)
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
from flow360.component.geometry_tree import (
    GeometryTree,
    NodeCollection,
    NodeType,
    TreeNode,
    TreeSearch,
)
from flow360.component.interfaces import GeometryInterface
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.folder import Folder
from flow360.component.simulation.primitives import Edge, GeometryBodyGroup, Surface
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.utils import (
    GeometryFiles,
    MeshNameParser,
    shared_account_confirm_proceed,
)
from flow360.exceptions import Flow360FileError, Flow360ValueError
from flow360.log import log


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


class GeometryDraft(ResourceDraft):
    """
    Geometry Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: Union[List[str], str],
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        folder: Optional[Folder] = None,
    ):
        self._file_names = file_names
        self.project_name = project_name
        self.tags = tags if tags is not None else []
        self.length_unit = length_unit
        self.solver_version = solver_version
        self.folder = folder
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_geometry()

    def _validate_geometry(self):
        if not isinstance(self.file_names, list) or len(self.file_names) == 0:
            raise Flow360FileError("file_names field has to be a non-empty list.")

        try:
            GeometryFiles(file_names=self.file_names)
        except pd.ValidationError as e:
            raise Flow360FileError(str(e)) from e

        for geometry_file in self.file_names:
            if not os.path.exists(geometry_file):
                raise Flow360FileError(f"{geometry_file} not found.")

        if self.project_name is None:
            self.project_name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
            log.warning(
                "`project_name` is not provided. "
                f"Using the first geometry file name {self.project_name} as project name."
            )

        if self.length_unit not in LengthUnitType.__args__:
            raise Flow360ValueError(
                f"specified length_unit : {self.length_unit} is invalid. "
                f"Valid options are: {list(LengthUnitType.__args__)}"
            )

        if self.solver_version is None:
            raise Flow360ValueError("solver_version field is required.")

    @property
    def file_names(self) -> List[str]:
        """geometry file"""
        if isinstance(self._file_names, str):
            return [self._file_names]
        return self._file_names

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(self, description="", progress_callback=None, run_async=False) -> Geometry:
        """
        Submit geometry to cloud and create a new project

        Parameters
        ----------
        description : str, optional
            description of the project, by default ""
        progress_callback : callback, optional
            Use for custom progress bar, by default None
        run_async : bool, optional
            Whether to submit Geometry asynchronously (default is False).

        Returns
        -------
        Geometry
            Geometry object with id
        """

        self._validate()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")
        mapbc_files = []
        for file_path in self.file_names:
            mesh_parser = MeshNameParser(file_path)
            if mesh_parser.is_ugrid() and os.path.isfile(
                mesh_parser.get_associated_mapbc_filename()
            ):
                file_name_mapbc = mesh_parser.get_associated_mapbc_filename()
                mapbc_files.append(file_name_mapbc)

        # Files with 'main' type are treated as MASTER_FILES and are processed after uploading
        # 'dependency' type files are uploaded only but not processed.
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

        ##:: Create new Geometry resource and project
        resp = RestApi(GeometryInterface.endpoint).post(req.dict())
        info = GeometryMeta(**resp)

        ##:: upload geometry files
        geometry = Geometry(info.id)
        heartbeat_info = {"resourceId": info.id, "resourceType": "Geometry", "stop": False}
        # Keep posting the heartbeat to keep server patient about uploading.
        heartbeat_thread = threading.Thread(target=post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()
        for file_path in self.file_names + mapbc_files:
            geometry._webapi._upload_file(
                remote_file_name=os.path.basename(file_path),
                file_name=file_path,
                progress_callback=progress_callback,
            )
        heartbeat_info["stop"] = True
        heartbeat_thread.join()
        ##:: kick off pipeline
        geometry._webapi._complete_upload()
        log.info(f"Geometry successfully submitted: {geometry.short_description()}")
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        if run_async:
            return geometry
        log.info("Waiting for geometry to be processed.")
        # uses from_cloud to ensure all metadata is ready before yielding the object
        return Geometry.from_cloud(info.id)


class FaceGroup:
    """
    Represents a face group that can be incrementally built by adding nodes.
    
    This class is returned by Geometry.create_face_group() and provides
    an .add() method to add more TreeNode instances to the group.
    
    FaceGroup instances are maintained by the parent Geometry object.
    """

    def __init__(self, geometry: "Geometry", name: str):
        """
        Initialize a FaceGroup.
        
        Parameters
        ----------
        geometry : Geometry
            The parent Geometry object that maintains this group
        name : str
            The name of this face group
        """
        self._geometry = geometry
        self._name = name
        self._faces: List[TreeNode] = []

    @property
    def name(self) -> str:
        """Get the name of this face group"""
        return self._name

    @property
    def faces(self) -> List[TreeNode]:
        """Get the list of face nodes in this group"""
        return self._faces

    @property
    def face_count(self) -> int:
        """Get the number of faces in this group"""
        return len(self._faces)

    def add(
        self, selection: Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
    ) -> "FaceGroup":
        """
        Add more nodes to this face group.
        
        This method delegates to the parent Geometry object to handle the addition.
        
        Parameters
        ----------
        selection : Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
            Nodes to add to this group. Can be:
            - TreeSearch instance - will be executed internally
            - NodeCollection - nodes will be extracted
            - Single TreeNode - will be wrapped in a list
            - List of TreeNode instances
            
        Returns
        -------
        FaceGroup
            Returns self for method chaining
            
        Examples
        --------
        >>> wing_group = geometry.create_face_group(name="wing", selection=...)
        >>> wing_group.add(geometry.tree_root.search(type=NodeType.FRMFeature, name="flap"))
        >>> wing_group.add(another_node)
        """
        # Delegate to Geometry to handle the addition
        self._geometry._add_to_face_group(self, selection)
        return self

    def __repr__(self):
        return f"FaceGroup(name='{self._name}', faces={self.face_count})"


class Geometry(AssetBase):
    """
    Geometry component for workbench (simulation V2)
    """

    _interface_class = GeometryInterface
    _meta_class = GeometryMeta
    _draft_class = GeometryDraft
    _web_api_class = Flow360Resource
    _entity_info_class = GeometryEntityInfo
    _cloud_resource_type_name = "Geometry"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tree: Optional[GeometryTree] = None
        self._face_groups: Dict[str, FaceGroup] = {}  # group_name -> FaceGroup instance
        self._face_uuid_to_face_group: Dict[str, FaceGroup] = {}  # face_uuid -> FaceGroup instance

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

        if self._entity_info is not None:
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
        cls, geometry_id: str = None, local_storage_path="", meta_data: GeometryMeta = None, allow_missing_entity_info = False 
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
            asset_id=geometry_id, local_storage_path=local_storage_path, meta_data=meta_data, allow_missing_entity_info = allow_missing_entity_info
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

    def reset_face_grouping(self) -> None:
        """Reset the face grouping"""
        # pylint: disable=protected-access,no-member
        self.internal_registry = self._entity_info._reset_grouping("face", self.internal_registry)

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

    # ========== Tree-based face grouping methods ==========

    @property
    def tree_root(self):
        """
        Get the root node of the geometry tree
        
        Returns
        -------
        TreeNode
            Root node of the geometry hierarchy tree
            
        Raises
        ------
        Flow360ValueError
            If geometry tree has not been loaded yet
        """
        if self._tree is None:
            raise Flow360ValueError(
                "Geometry tree not loaded. Call load_geometry_tree() first with path to tree.json"
            )
        return self._tree.root

    def load_geometry_tree(self, tree_json_path: str) -> None:
        """
        Load Geometry hierarchy tree from JSON file

        Parameters
        ----------
        tree_json_path : str
            Path to the tree JSON file generated from Geometry hierarchy extraction

        Examples
        --------
        >>> geometry = Geometry.from_cloud("geom_id")
        >>> geometry.load_geometry_tree("tree.json")
        """
        self._tree = GeometryTree(tree_json_path)
        ## create default face gruoping  by body

        log.info(f"Loaded Geometry tree with {len(self._tree.all_faces)} faces")

        body_nodes = self.tree_root.search(type = NodeType.RiBrepModel).execute()
        print("abc: All body nodes:")
        for body_node in body_nodes:
            print(body_node)
            self.create_face_group(
                name = body_node.name,
                selection = body_node,
            )

        print("abc: After the default face grouping by body is finished: ")
        self.print_face_grouping_stats()

    def create_face_group(
        self, name: str, selection: Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
    ) -> FaceGroup:
        """
        Create a face group based on explicit selection of tree nodes

        This method groups all faces under the selected nodes in the Geometry hierarchy tree.
        If any faces already belong to another group, they will be reassigned to the new group.
        
        Returns a FaceGroup object that can be used to incrementally add more nodes.

        Parameters
        ----------
        name : str
            Name of the face group
        selection : Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
            Can be one of:
            - TreeSearch instance (returned from tree_root.search()) - will be executed internally
            - NodeCollection (returned from tree_root.children()) - nodes will be extracted
            - Single TreeNode - all faces under this node will be included
            - List of TreeNode instances - all faces under these nodes will be included
            
            All faces under the selected nodes (recursively) will be added to the group.

        Returns
        -------
        FaceGroup
            A FaceGroup object that can be used to add more nodes via .add() method

        Examples
        --------
        >>> from flow360.component.geometry_tree import NodeType
        >>> 
        >>> # Using TreeSearch (recommended - captures intent declaratively)
        >>> wing_group = geometry.create_face_group(
        ...     name="wing",
        ...     selection=geometry.tree_root.search(type=NodeType.FRMFeature, name="*wing*")
        ... )
        >>> 
        >>> # Using children() chaining (fluent navigation with exact matching)
        >>> body_group = geometry.create_face_group(
        ...     name="body",
        ...     selection=geometry.tree_root.children().children().children(
        ...         type=NodeType.FRMFeatureBasedEntity
        ...     ).children().children(type=NodeType.FRMFeature, name="body_main")
        ... )
        >>> 
        >>> # Incrementally add more nodes to the group
        >>> body_group.add(
        ...     geometry.tree_root.children().children().children(
        ...         type=NodeType.FRMFeatureBasedEntity
        ...     ).children().children(type=NodeType.FRMFeature, name="body_cut")
        ... )
        """
        if self._tree is None:
            raise Flow360ValueError(
                "Geometry tree not loaded. Call load_geometry_tree() first with path to tree.json"
            )

        # Get or create FaceGroup
        if name not in self._face_groups:
            face_group = FaceGroup(self, name)
            self._face_groups[name] = face_group
            log.info(f"Created face group '{name}'")
        else:
            face_group = self._face_groups[name]
            log.info(f"Using existing face group '{name}'")

        # Add faces to the group
        self._add_to_face_group(face_group, selection)
        
        return face_group

    def _add_to_face_group(
        self, face_group: FaceGroup, selection: Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
    ) -> None:
        """
        Internal method to add faces to a face group.
        
        This method handles the core logic of:
        - Converting selection to a list of nodes
        - Extracting all faces from selected nodes
        - Removing faces from their previous groups
        - Adding faces to the target group
        - Updating the face-to-group mapping
        - Automatically removing any face groups that become empty
        
        Parameters
        ----------
        face_group : FaceGroup
            The face group to add faces to
        selection : Union[TreeNode, List[TreeNode], NodeCollection, TreeSearch]
            The selection to add (TreeSearch, NodeCollection, TreeNode, or list of TreeNodes)
        
        Notes
        -----
        If moving faces causes any group to become empty (0 faces), that group will be
        automatically removed from the Geometry's face group registry.
        """
        # Handle different selection types
        if isinstance(selection, TreeSearch):
            selected_nodes = selection.execute()
        elif isinstance(selection, NodeCollection):
            selected_nodes = selection.nodes
        elif isinstance(selection, TreeNode):
            selected_nodes = [selection]
        else:
            selected_nodes = selection

        # Collect faces from selected nodes
        new_faces = []
        new_face_uuids = set()
        
        for node in selected_nodes:
            faces = node.get_all_faces()
            for face in faces:
                if face.uuid:
                    new_faces.append(face)
                    new_face_uuids.add(face.uuid)

        # Remove these faces from their previous groups
        groups_to_check = set()
        for uuid in new_face_uuids:
            if uuid in self._face_uuid_to_face_group:
                old_group = self._face_uuid_to_face_group[uuid]
                if old_group != face_group:
                    # Remove from old group
                    old_group._faces = [f for f in old_group._faces if f.uuid != uuid]
                    groups_to_check.add(old_group)

        # Clean up empty face groups
        for group in groups_to_check:
            if len(group._faces) == 0:
                # Remove the empty group from the registry
                if group.name in self._face_groups:
                    del self._face_groups[group.name]
                    log.info(f"Removed empty face group '{group.name}'")

        # Update face-to-group mapping
        for uuid in new_face_uuids:
            self._face_uuid_to_face_group[uuid] = face_group

        # Add to this group (avoiding duplicates)
        existing_uuids = {f.uuid for f in face_group._faces if f.uuid}
        
        added_count = 0
        for face in new_faces:
            if face.uuid not in existing_uuids:
                face_group._faces.append(face)
                existing_uuids.add(face.uuid)
                added_count += 1

        log.info(
            f"Added {added_count} faces to group '{face_group.name}' "
            f"(total: {len(face_group._faces)} faces)"
        )

    def face_grouping_configuration(self) -> Dict[str, str]:
        face_uuid_to_face_group_name = {}
        for face_uuid, face_group in self._face_uuid_to_face_group.items():
            face_uuid_to_face_group_name[face_uuid] = face_group.name
        return face_uuid_to_face_group_name

    def print_face_grouping_stats(self) -> None:
        """
        Print statistics about face grouping

        Examples
        --------
        >>> geometry.print_face_grouping_stats()
        === Face Grouping Statistics ===
        Total faces: 95
        Faces in groups: 95

        Face groups (3):
          - wing: 45 faces
          - fuselage: 32 faces
          - tail: 18 faces
        =================================
        """
        if self._tree is None:
            raise Flow360ValueError(
                "Geometry tree not loaded. Call load_geometry_tree() first with path to tree.json"
            )
        
        total_faces = len(self._tree.all_faces)
        faces_in_groups = sum(group.face_count for group in self._face_groups.values())

        print(f"\n=== Face Grouping Statistics ===")
        print(f"Total faces: {total_faces}")
        print(f"\nFace groups ({len(self._face_groups)}):")
        for group_name, group in self._face_groups.items():
            print(f"  - {group_name}: {group.face_count} faces")
        print("="*33)
