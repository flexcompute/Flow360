"""
Geometry component
"""

from __future__ import annotations

import os
from typing import List, Union

import pydantic.v1 as pd

from flow360.cloud.requests_v2 import (
    GeometryFileMeta,
    LengthUnitType,
    NewGeometryRequest,
)
from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.interfaces import GeometryInterface
from flow360.component.simulation.web.resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    ResourceDraft,
)
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    match_file_pattern,
    shared_account_confirm_proceed,
)
from flow360.component.volume_mesh import VolumeMesh
from flow360.exceptions import Flow360FileError, Flow360ValueError
from flow360.log import log


# pylint: disable=R0801
class GeometryMeta(Flow360ResourceBaseModel):
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
        name: str = None,
        tags: List[str] = None,
        length_unit: LengthUnitType = "m",
    ):
        self._file_names = file_names
        self.name = name
        self.tags = tags if tags is not None else []
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
        if self.length_unit not in LengthUnitType.__args__:
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
            # pylint: disable=fixme
            # TODO: remove hardcoding
            parent_folder_id="ROOT.FLOW360",
            length_unit=self.length_unit,
            description=description,
        )

        ##:: Create new Geometry resource and project
        resp = RestApi(GeometryInterface.endpoint).post(req.model_dump(by_alias=True))
        info = GeometryMeta(**resp)

        ##:: upload geometry files
        geometry = Geometry(info.id)
        for file_path in self.file_names:
            file_name = os.path.basename(file_path)
            geometry._web._upload_file(
                remote_file_name=file_name, file_name=file_path, progress_callback=progress_callback
            )

        ##:: kick off pipeline
        RestApi(GeometryInterface.endpoint, id=geometry._web.id).patch(
            {"action": "Success"}, method="files"
        )
        log.info("Geometry successfully submitted.")
        return geometry


class Geometry(AssetBase):
    """
    Geometry component for workbench (simulation V2)
    """

    _interface = GeometryInterface
    _info_type_class = GeometryMeta
    _draft_class = GeometryDraft
    _web: Flow360Resource = None

    @classmethod
    def from_file(
        cls,
        file_names: Union[List[str], str],
        name: str = None,
        tags: List[str] = None,
        length_unit: LengthUnitType = "m",
    ) -> GeometryDraft:
        # For type hint only but proper fix is to fully abstract the Draft class too.
        return super().from_file(file_names, name, tags, length_unit)

    def _retrieve_metadata(self):
        # get the metadata when initializing the object (blocking)
        # My next PR
        pass

    def generate_surface_mesh(self, params: SimulationParams, async_mode: bool = True):
        """generate surface mesh from the geometry"""
        return self.run(params, SurfaceMesh, async_mode)

    def generate_volume_mesh(self, params: SimulationParams, async_mode: bool = True):
        """generate volume mesh from the geometry"""
        return self.run(params, VolumeMesh, async_mode)

    def run_case(self, params: SimulationParams, async_mode: bool = True):
        """run case from the geometry"""
        return self.run(params, Case, async_mode)
