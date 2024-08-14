"""
Geometry component
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Union

import pydantic.v1 as pd

from flow360.cloud.requests import GeometryFileMeta, LengthUnitType, NewGeometryRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import GeometryInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    match_file_pattern,
    shared_account_confirm_proceed,
)
from flow360.exceptions import Flow360FileError, Flow360ValueError
from flow360.log import log

HEARTBEAT_INTERVAL = 15


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


# pylint: disable=R0801
class GeometryMeta(AssetMetaBaseModel):
    """
    GeometryMeta component
    """

    project_id: str = pd.Field(alias="projectId")


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
        log.info("Geometry successfully submitted.")
        return geometry


class Geometry(AssetBase):
    """
    Geometry component for workbench (simulation V2)
    """

    _interface_class = GeometryInterface
    _info_type_class = GeometryMeta
    _draft_class = GeometryDraft
    _webapi: Flow360Resource = None

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

    def _get_metadata(self):
        # get the metadata when initializing the object (blocking)
        # My next PR
        pass
