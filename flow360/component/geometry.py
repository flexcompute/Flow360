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
    NewGeometryRequest,
)
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
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
                file_name_mapbc = os.path.basename(mesh_parser.get_associated_mapbc_filename())
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

    def get_default_settings(self, simulation_dict: dict):
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

        self.default_settings["geometry_accuracy"] = _get_default_geometry_accuracy(
            simulation_dict=simulation_dict
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
