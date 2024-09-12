"""
Geometry component
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic as pd

from flow360.cloud.requests import GeometryFileMeta, LengthUnitType, NewGeometryRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import GeometryInterface
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import Edge, Surface
from flow360.component.simulation.utils import _model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import _get_simulation_json_from_cloud
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    match_file_pattern,
    shared_account_confirm_proceed,
    validate_type,
)
from flow360.exceptions import Flow360FileError, Flow360ValueError, Flow360RuntimeError
from flow360.log import log

HEARTBEAT_INTERVAL = 15
TIMEOUT_MINUTES = 30


def _post_upload_heartbeat(info):
    """
    Keep letting the server know that the uploading is still in progress.
    Server marks resource as failed if no heartbeat is received for 3 `heartbeatInterval`s.
    """
    while not info["stop"]:
        RestApi("v2/heartbeats/uploading").post(
            {
                "resourceId": info["resourceId"],
                "heartbeatInterval": HEARTBEAT_INTERVAL,
                "resourceType": info["resourceType"],
            }
        )
        time.sleep(HEARTBEAT_INTERVAL)


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

    project_id: str = pd.Field(alias="projectId")
    deleted: bool = pd.Field()
    entity_info: Optional[GeometryEntityInfo] = pd.Field(None)
    status: GeometryStatus = pd.Field()  # Overshadowing to ensure correct is_final() method


class GeometryWebAPI(Flow360Resource):
    """web API for Geometry resource. This can and should be generalized and reused by all resources."""

    def get_entity_info(self, force: bool = False):
        """
        Blockingly trying to download the entityInfo.json
        """
        self._info = super().get_info()
        if getattr(self._info, "metadata", None) is not None and force is False:
            log.debug("Metadata already loaded. Skipping download.")
            return

        start_time = time.time()
        while self.status.is_final() is False:
            if time.time() - start_time > TIMEOUT_MINUTES * 60:
                raise TimeoutError(
                    "Timeout: Process did not finish within the specified timeout period"
                )
            time.sleep(2)

        log.debug("Metadata pipeline completed, downloading metadata now...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            # Windows OS complains when a file is opened in write mode and then read mode. So we need to close it first.
            # pylint: disable=protected-access
            self._download_file(
                "simulation.json",
                to_file=temp_file.name,
                to_folder=".",
                overwrite=True,
                progress_callback=None,
                verbose=False,
            )
            temp_file.flush()
            temp_file_name = temp_file.name
            temp_file.close()

        try:
            with open(temp_file_name, "r", encoding="utf-8") as f:
                _meta = json.load(f)
                if "private_attribute_asset_cache" in _meta.keys() and "project_entity_info" in _meta['private_attribute_asset_cache'].keys():            
                    _meta = _meta['private_attribute_asset_cache']['project_entity_info']
                else:
                    raise Flow360RuntimeError('"[Internal Error] processing geometry failed."')

                # pylint: disable=protected-access
                self._info = self._info.model_copy(
                    deep=True, update={"entity_info": GeometryEntityInfo(**_meta)}
                )
                assert isinstance(
                    self._info.entity_info, GeometryEntityInfo
                ), "[Internal Error] Entity info parsing failed."
        finally:
            os.remove(temp_file_name)
        log.debug("Entity info loaded successfully.")

    @classmethod
    def _from_meta(cls, meta: GeometryMeta) -> GeometryWebAPI:
        validate_type(meta, "meta", GeometryMeta)
        geometry_web_api = cls(id=meta.id, interface=GeometryInterface, meta_class=GeometryMeta)
        geometry_web_api._set_meta(meta)
        return geometry_web_api


class GeometryDraft(ResourceDraft):
    """
    Geometry Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: List[str],
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        self._file_names = file_names
        self.project_name = project_name
        self.tags = tags if tags is not None else []
        self.length_unit = length_unit
        self.solver_version = solver_version
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_geometry()

    def _validate_geometry(self):

        if not isinstance(self.file_names, list) or len(self.file_names) == 0:
            raise Flow360FileError("file_names field has to be a non-empty list.")

        for geometry_file in self.file_names:
            _, ext = os.path.splitext(geometry_file)
            if not match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, geometry_file):
                raise Flow360FileError(
                    f"Unsupported geometry file extensions: {ext.lower()}. "
                    f"Supported: [{', '.join(SUPPORTED_GEOMETRY_FILE_PATTERNS)}]."
                )

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
        return self._file_names

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(self, description="", progress_callback=None) -> Geometry:
        """
        Submit geometry to cloud and create a new project

        Parameters
        ----------
        description : str, optional
            description of the project, by default ""
        progress_callback : callback, optional
            Use for custom progress bar, by default None

        Returns
        -------
        Geometry
            Geometry object with id
        """

        self._validate()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")
        # The first geometry is assumed to be the main one.
        req = NewGeometryRequest(
            name=self.project_name,
            solver_version=self.solver_version,
            tags=self.tags,
            files=[
                GeometryFileMeta(
                    name=os.path.basename(file_path),
                    type="main" if item_index == 0 else "dependency",
                )
                for item_index, file_path in enumerate(self.file_names)
            ],
            # pylint: disable=fixme
            # TODO: remove hardcoding
            parent_folder_id="ROOT.FLOW360",
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
        heartbeat_thread = threading.Thread(target=_post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()
        for file_path in self.file_names:
            geometry._webapi._upload_file(
                remote_file_name=os.path.basename(file_path),
                file_name=file_path,
                progress_callback=progress_callback,
            )
        heartbeat_info["stop"] = True
        heartbeat_thread.join()
        ##:: kick off pipeline
        geometry._webapi._complete_upload()
        log.debug("Waiting for geometry to be processed.")
        geometry._webapi.get_info()
        log.info("Geometry successfully submitted.")
        return geometry


class Geometry(AssetBase):
    """
    Geometry component for workbench (simulation V2)
    """

    _interface_class = GeometryInterface
    _meta_class = GeometryMeta
    _draft_class = GeometryDraft
    _web_api_class = GeometryWebAPI
    face_group_tag: str = None
    edge_group_tag: str = None

    @classmethod
    # pylint: disable=redefined-builtin
    def from_cloud(cls, id: str):
        """Create asset with the given ID"""
        asset_obj = super().from_cloud(id)
        # get the face tag and edge tag used.
        simulation_dict = _get_simulation_json_from_cloud(asset_obj.project_id)
        if "private_attribute_asset_cache" not in simulation_dict:
            raise KeyError(
                "[Internal] Could not find private_attribute_asset_cache in the draft's simulation settings."
            )
        asset_cache = simulation_dict["private_attribute_asset_cache"]

        if "private_attribute_asset_cache" not in asset_cache:
            raise KeyError(
                "[Internal] Could not find private_attribute_asset_cache in the draft's simulation settings."
            )
        entity_info = asset_cache["project_entity_info"]
        if "face_group_tag" not in entity_info or entity_info["face_group_tag"] is None:
            # This may happen if users submit the Geometry but did not do anything else.
            # Then they load back the geometry which will then have no info on grouping.
            log.warning(
                "Could not find face grouping info in the draft's simulation settings. "
                "Please remember to group them if relevant features are used."
            )
            asset_obj.face_group_tag = entity_info["face_group_tag"]
        if "edge_group_tag" not in entity_info or entity_info["edge_group_tag"] is None:
            log.warning(
                "Could not find face grouping info in the draft's simulation settings. "
                "Please remember to group them if relevant features are used."
            )
            asset_obj.edge_group_tag = entity_info["edge_group_tag"]
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
    ) -> GeometryDraft:
        # For type hint only but proper fix is to fully abstract the Draft class too.
        return super().from_file(file_names, project_name, solver_version, length_unit, tags)

    @property
    def entity_info(self):
        """Return the entity info of the resource"""
        # pylint: disable=protected-access
        self._webapi.get_entity_info()
        return self._webapi._info.entity_info

    def show_available_groupings(self, verbose_mode: bool = False):
        """Display all the possible groupings for faces and edges"""
        self._show_avaliable_entity_groups(
            "faces",
            ignored_attribute_tags=["__all__", "faceId"],
            show_ids_in_each_group=verbose_mode,
        )
        self._show_avaliable_entity_groups(
            "edges",
            ignored_attribute_tags=["__all__", "edgeId"],
            show_ids_in_each_group=verbose_mode,
        )

    def _show_avaliable_entity_groups(
        self,
        entity_type_name: Literal["faces", "edges"],
        ignored_attribute_tags: list = None,
        show_ids_in_each_group: bool = False,
    ) -> None:
        """
        Display all the grouping info for the given entity type
        """

        if entity_type_name not in ["faces", "edges"]:
            raise Flow360ValueError(
                f"entity_type_name: {entity_type_name} is invalid. Valid options are: ['faces', 'edges']"
            )

        log.info(f" >> Available attribute tags for grouping **{entity_type_name}**:")
        # pylint: disable=no-member
        if entity_type_name == "faces":
            attribute_names = self.entity_info.face_attribute_names
            grouped_items = self.entity_info.grouped_faces
        else:
            attribute_names = self.entity_info.edge_attribute_names
            grouped_items = self.entity_info.grouped_edges
        for tag_index, attribute_tag in enumerate(attribute_names):
            if ignored_attribute_tags is not None and attribute_tag in ignored_attribute_tags:
                continue
            log.info(f"    >> Tag {tag_index}: {attribute_tag}")
            for index, entity in enumerate(grouped_items[tag_index]):
                log.info(f"        >> Group {index}: {entity.name}")
                if show_ids_in_each_group is True:
                    log.info(f"           IDs: {entity.private_attribute_sub_components}")

    def _group_entity_by_tag(
        self, entity_type_name: Literal["face", "edge"], tag_name: str
    ) -> None:
        if hasattr(self, "internal_registry") is False or self.internal_registry is None:
            self.internal_registry = EntityRegistry()

        found_existing_grouping = (
            self.face_group_tag is not None
            if entity_type_name == "face"
            else self.edge_group_tag is not None
        )
        if found_existing_grouping is True:
            # pylint: disable=fixme
            # TODO: We need to make sure only 1 grouping is used in simluationParams.
            log.warning(
                f"Grouping already exists for {entity_type_name}. Resetting the grouping and regroup with {tag_name}."
            )
            self._reset_grouping(entity_type_name)

        self.internal_registry = self.entity_info.group_in_registry(
            entity_type_name, attribute_name=tag_name, registry=self.internal_registry
        )
        if entity_type_name == "face":
            self.face_group_tag = tag_name
        else:
            self.edge_group_tag = tag_name

    def group_faces_by_tag(self, tag_name: str) -> None:
        """
        Group faces by tag name
        """
        self._group_entity_by_tag("face", tag_name)

    def group_edges_by_tag(self, tag_name: str) -> None:
        """
        Group edges by tag name
        """
        self._group_entity_by_tag("edge", tag_name)

    def _reset_grouping(self, entity_type_name: Literal["face", "edge"]) -> None:
        if entity_type_name == "face":
            self.internal_registry.clear(Surface)
        else:
            self.internal_registry.clear(Edge)

        if entity_type_name == "face":
            self.face_group_tag = None
        else:
            self.edge_group_tag = None

    def reset_face_grouping(self) -> None:
        """Reset the face grouping"""
        self._reset_grouping("face")

    def reset_edge_grouping(self) -> None:
        """Reset the edge grouping"""
        self._reset_grouping("edge")

    def __getitem__(self, key: str):
        """
        Get the entity by name.
        `key` is the name of the entity or the naming pattern if wildcard is used.
        """
        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        if hasattr(self, "internal_registry") is False or self.internal_registry is None:
            raise Flow360ValueError(
                "The faces/edges in geometry are not grouped yet."
                "Please use `group_faces_by_tag` or `group_edges_by_tag` function to group them first."
            )
            # Note: Or we assume group default by just FaceID and EdgeID? Not sure if this is actually useful.
        return self.internal_registry.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError("Assigning/setting entities is not supported.")

    def _inject_entity_info_to_params(self, params):
        params = super()._inject_entity_info_to_params(params)
        with _model_attribute_unlock(
            params.private_attribute_asset_cache.project_entity_info, "face_group_tag"
        ):
            params.private_attribute_asset_cache.project_entity_info.face_group_tag = (
                self.face_group_tag
            )
        with _model_attribute_unlock(
            params.private_attribute_asset_cache.project_entity_info, "edge_group_tag"
        ):
            params.private_attribute_asset_cache.project_entity_info.edge_group_tag = (
                self.edge_group_tag
            )
        return params