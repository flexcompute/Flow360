"""
Geometry component
"""

from __future__ import annotations

import os
from typing import List, Union, Literal
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.surface_mesh import SurfaceMesh
import pydantic.v1 as pd

from ..cloud.rest_api import RestApi
from ..exceptions import Flow360FileError, Flow360ValueError
from ..log import log
from .interfaces import GeometryInterface, ProjectInterface, DraftInterface
from .resource_base import Flow360Resource, Flow360ResourceBaseModel, ResourceDraft
from .utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    match_file_pattern,
    shared_account_confirm_proceed,
    validate_type,
)
import time
from flow360.cloud.requests_v2 import NewGeometryRequest, GeometryFileMeta, length_unit_type


class Geometry:
    """
    Geometry component for workbench (simulation V2)
    """

    _api: Flow360Resource = None

    def _retrieve_metadata(self):
        # get the metadata when initializing the object (blocking)
        pass

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        self._api = Flow360Resource(
            interface=GeometryInterface,
            info_type_class=GeometryMeta,
            id=id,
        )
        self._retrieve_metadata()
        self._params = None
        # get the project id according to geometry id
        resp = RestApi(GeometryInterface.endpoint, id=id).get()
        project_id = resp["projectId"]
        solver_version = resp["solverVersion"]
        print(">> project_id = ", project_id)
        self.project_id = project_id
        self.solver_version = solver_version

    @classmethod
    def _from_meta(cls, meta: GeometryMeta):
        validate_type(meta, "meta", GeometryMeta)
        geometry = cls(id=meta.id)
        geometry._api._set_meta(meta)
        return geometry

    @property
    def info(self) -> GeometryMeta:
        return self._api.info

    @classmethod
    def _interface(cls):
        return GeometryInterface

    @classmethod
    def _meta_class(cls):
        """
        returns geometry meta info class: GeometryMeta
        """
        return GeometryMeta

    @classmethod
    def from_cloud(cls, geometry_id: str):
        """
        Get geometry from cloud
        :param geometry_id:
        :return:
        """
        return cls(geometry_id)

    @classmethod
    def from_file(
        cls,
        file_names: Union[List[str], str],
        name: str = None,
        tags: List[str] = None,
        length_unit: length_unit_type = "m",
    ):
        """
        Create geometry from geometry files
        :param file_names:
        :param name:
        :param tags:
        :return:
        """
        if isinstance(file_names, str):
            file_names = [file_names]
        return GeometryDraft(
            file_names=file_names,
            name=name,
            tags=tags,
            length_unit=length_unit,
        )

    def generate_surface_mesh(
        self,
        params: SimulationParams,
        async_mode=True,
        timeout_minutes: int = 60,
        progress_callback=None,
    ) -> SurfaceMesh:
        """
        Generate surface mesh with given simulation params.
        async_mode: if True, returns SurfaceMesh object immediately, otherwise waits for the meshing to finish.
        """
        ##:: Get the latest draft of the project:
        draft_id = RestApi(ProjectInterface.endpoint, id=self.project_id).get()["lastOpenDraftId"]
        if draft_id is None:  # No saved online session
            ##:: Get new draft id
            draft_id = RestApi(DraftInterface.endpoint).post(
                {
                    "name": "",  # TODO: add time to differentiate
                    "projectId": self.project_id,
                    "sourceItemId": self._api.id,
                    "sourceItemType": "Geometry",
                    "solverVersion": self.solver_version,
                    "forkCase": False,
                }
            )["id"]
        ##:: Post the simulation param:
        req = {"data": params.model_dump_json(), "type": "simulation", "version": ""}
        RestApi(DraftInterface.endpoint, id=draft_id).post(json=req, method="simulation/file")
        ##:: Kick off draft run:
        surface_mesh_id = RestApi(DraftInterface.endpoint, id=draft_id).post(
            json={"upTo": "SurfaceMesh", "useInHouse": True}, method="run"
        )["id"]
        ##:: Patch project
        RestApi(ProjectInterface.endpoint, id=self.project_id).patch(
            json={
                "lastOpenItemId": surface_mesh_id,
                "lastOpenItemType": "SurfaceMesh",
            }
        )
        surface_mesh = SurfaceMesh(surface_mesh_id)
        if async_mode:
            return surface_mesh
        else:
            start_time = time.time()
            while surface_mesh.status.is_final() == False:
                if time.time() - start_time > timeout_minutes * 60:
                    raise TimeoutError(
                        "Timeout: Process did not finish within the specified timeout period"
                    )
                log.info(">>> Processing...")
                time.sleep(10)


class GeometryMeta(Flow360ResourceBaseModel):
    """
    GeometryMeta component
    """

    project_id: str = pd.Field(alias="projectId")

    def to_geometry(self) -> Geometry:
        """
        returns Geometry object from geometry meta info
        """
        return Geometry(self.id)


class GeometryDraft(ResourceDraft):
    """
    Geometry Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: List[str],
        name: str = None,
        tags: List[str] = None,
        length_unit: length_unit_type = "m",
    ):
        self._file_names = file_names
        self.name = name
        self.tags = tags if tags is not None else []
        self._id = None
        self.length_unit = length_unit
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_geometry()

    # pylint: disable=consider-using-f-string
    def _validate_geometry(self):
        if not isinstance(self.file_names, list):
            raise Flow360FileError("file_names field has to be a list.")
        for geometry_file in self.file_names:
            _, ext = os.path.splitext(geometry_file)
            if not match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, geometry_file):
                raise Flow360FileError(
                    "Unsupported geometry file extensions: {}. Supported: [{}].".format(
                        ext.lower(), ", ".join(SUPPORTED_GEOMETRY_FILE_PATTERNS)
                    )
                )

            if not os.path.exists(geometry_file):
                raise Flow360FileError(f"{geometry_file} not found.")

        if self.name is None and len(self.file_names) > 1:
            raise Flow360ValueError(
                "name field is required if more than one geometry files are provided."
            )
        if self.length_unit not in length_unit_type.__args__:
            raise Flow360ValueError(f"specified length_unit : {self.length_unit} is invalid.")

    @property
    def file_names(self) -> List[str]:
        """geometry file"""
        return self._file_names

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(self, solver_version, description="", progress_callback=None) -> Geometry:
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
        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
        self.name = name

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")
        # The first geometry is assumed to be the main one.
        req = NewGeometryRequest(
            name=self.name,
            solver_version=solver_version,
            tags=self.tags,
            files=[
                GeometryFileMeta(
                    name=os.path.basename(file_path),
                    type="main" if item_index == 0 else "dependency",
                )
                for item_index, file_path in enumerate(self.file_names)
            ],
            # files=[GeometryFileMeta(name="geometry.csm", type="main")],
            parent_folder_id="ROOT.FLOW360",  # TODO: remove hardcoding
            length_unit=self.length_unit,
            description=description,
        )

        ##:: Create new Geometry resource and project
        # TODO: we need project id?
        resp = RestApi(GeometryInterface.endpoint).post(req.model_dump(by_alias=True))
        info = GeometryMeta(**resp)
        self._id = info.id
        print("resp project_id = ", info.project_id)
        self.solver_version = solver_version

        ##:: upload geometry files
        _temp_resource_handler = Flow360Resource(
            interface=GeometryInterface,
            info_type_class=GeometryMeta,
            id=info.id,
        )
        for file_path in self.file_names:
            file_name = os.path.basename(file_path)
            _temp_resource_handler._upload_file(
                remote_file_name=file_name, file_name=file_path, progress_callback=progress_callback
            )

        ##:: kick off pipeline
        RestApi(GeometryInterface.endpoint, id=_temp_resource_handler.id).patch(
            {"action": "Success"}, method="files"
        )

        log.info(f"Geometry successfully submitted.")
        return Geometry(self.id)
