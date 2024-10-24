import json
from typing import Optional

import pydantic as pd

from enum import Enum

from flow360 import SurfaceMesh, Case
from flow360.cloud.requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.geometry import Geometry
from flow360.component.interfaces import ProjectInterface, GeometryInterface, VolumeMeshInterfaceV2
from flow360.component.simulation import cloud
from flow360.component.utils import SUPPORTED_GEOMETRY_FILE_PATTERNS, match_file_pattern, MeshNameParser
from flow360.component.volume_mesh import VolumeMesh
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.exceptions import Flow360ValueError, Flow360FileError, Flow360WebError


class RootType(Enum):
    GEOMETRY = "Geometry"
    VOLUME_MESH = "Volume mesh"
    # SURFACE_MESH = "Surface mesh" - supported in a future iteration


class ProjectMeta(pd.BaseModel, extra=pd.Extra.allow):
    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    name: str = pd.Field()
    root_item_id: str = pd.Field(alias="rootItemId")
    root_item_type: RootType = pd.Field(alias="rootItemType")


class Project(pd.BaseModel):
    info: ProjectMeta = pd.Field(None)

    # Right now the cached assets are all related to the
    # root asset - we will have full project tree traversal
    # in a future iteration.
    _geometry: Optional[Geometry] = pd.PrivateAttr(None)
    _volume_mesh: Optional[VolumeMesh] = pd.PrivateAttr(None)
    _surface_mesh: Optional[SurfaceMesh] = pd.PrivateAttr(None)
    _case: Optional[Case] = pd.PrivateAttr(None)

    _params: Optional[SimulationParams] = pd.PrivateAttr(None)

    @property
    def geometry(self) -> Geometry:
        """
        Getter for the current project's geometry asset

        Returns: Geometry asset if present, otherwise raises Flow360ValueError
        """
        if not self._geometry:
            raise Flow360ValueError("Geometry asset is not available for this project")

        return self._geometry

    @property
    def surface_mesh(self) -> SurfaceMesh:
        """
        Getter for the current project's surface mesh asset

        Returns: Surface mesh asset if present, otherwise raises Flow360ValueError
        """
        if not self._surface_mesh:
            raise Flow360ValueError("Surface mesh asset is not available for this project")

        return self._surface_mesh

    @property
    def volume_mesh(self) -> VolumeMesh:
        """
        Getter for the current project's volume mesh asset

        Returns: Volume mesh asset if present, otherwise raises Flow360ValueError
        """
        if not self._volume_mesh:
            raise Flow360ValueError("Volume mesh asset is not available for this project")

        return self._volume_mesh

    @property
    def case(self) -> Case:
        """
        Getter for the current project's case asset

        Returns: Case asset if present, otherwise raises Flow360ValueError
        """
        if not self._case:
            raise Flow360ValueError("Case asset is not available for this project")

        return self._case

    @classmethod
    def _detect_root_type(cls, file):
        if match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, file):
            return RootType.GEOMETRY
        else:
            try:
                parser = MeshNameParser(file)
                if parser.is_valid_volume_mesh():
                    return RootType.VOLUME_MESH
            except Flow360FileError:
                pass

        raise Flow360FileError(f"{file} is not a geometry or volume mesh file required for project initialization.")

    @classmethod
    def from_file(
        cls,
        file: str = None,
        name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m"
    ):
        """
        Populates project data from a file

        Args:
            name (): Name of the new project
            file ():
            solver_version ():
            length_unit ():

        Returns:

        """

        root_asset = None

        root_type = cls._detect_root_type(file)

        match root_type:
            case RootType.GEOMETRY:
                # Create a draft geometry asset and submit
                draft = Geometry.from_file(
                    file,
                    name,
                    solver_version,
                    length_unit
                )
                root_asset = draft.submit()
            case RootType.VOLUME_MESH:
                # Create a draft volume mesh asset and submit
                draft = VolumeMesh.from_file(
                    file,
                    name,
                    solver_version,
                )
                root_asset = draft.submit()

        if not root_asset:
            raise Flow360ValueError(f"Couldn't initialize asset from {file}")

        project_id = root_asset.project_id

        project_api = RestApi(ProjectInterface.endpoint, id=project_id)
        info = project_api.get()

        if not info:
            raise Flow360ValueError(f"Couldn't retrieve project info for {project_id}")

        project = Project(info=ProjectMeta(**info))

        match root_type:
            case RootType.GEOMETRY:
                project._geometry = root_asset
            case RootType.VOLUME_MESH:
                project._volume_mesh = root_asset

        return project

    @classmethod
    def from_cloud(cls, project_id: str):
        """Load project from cloud, only load the root asset for now"""
        project_api = RestApi(ProjectInterface.endpoint, id=project_id)
        info = project_api.get()

        if not isinstance(info, dict):
            raise Flow360WebError(f"Cannot load project {project_id}, missing project data.")

        if not info:
            raise Flow360WebError(f"Couldn't retrieve project info for {project_id}")

        project = Project(info=ProjectMeta(**info))

        root_type = project.info.root_item_type

        match root_type:
            case RootType.GEOMETRY:
                root_asset = Geometry.from_cloud(project.info.root_item_id)
                project._geometry = root_asset
            case RootType.VOLUME_MESH:
                root_asset = VolumeMesh.from_cloud(project.info.root_item_id)
                project._volume_mesh = root_asset

        return project

    def _check_initialized(self):
        if not self.info:
            raise Flow360ValueError("Project is not initialized - use Project.from_file or Project.from_cloud")

    def set_default_params(self, params: SimulationParams):
        self._params = params

    def get_simulation_json(self):
        """
        Get default simulation JSON for the project based on the root asset type
        """
        self._check_initialized()

        root_type = self.info.root_item_type
        root_id = self.info.root_item_id

        if not root_type or not root_id:
            raise Flow360ValueError("Root item type or ID is missing from project metadata")

        root_api = None

        match root_type:
            case RootType.GEOMETRY:
                root_api = RestApi(GeometryInterface.endpoint, id=root_id)
            case RootType.VOLUME_MESH:
                root_api = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_id)

        resp = root_api.get(method="simulation/file", params={"type": "simulation"})

        if not isinstance(resp, dict) or "simulationJson" not in resp:
            raise Flow360WebError("Root item type or ID is missing from project metadata")

        simulation_json = json.loads(resp["simulationJson"])

        return simulation_json

    def run_surface_mesher(self, params: SimulationParams = None, draft_name: str = "Surface Mesh"):
        """Run surface mesher with the provided params or defaults"""
        self._check_initialized()

        if self.info.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError("Surface mesher can only be ran by projects with a geometry root asset")
        if self._geometry is None:
            raise Flow360ValueError("This project object was not initialized using from_cloud or from_file")
        # We should remove usages of cloud static functions here
        self._surface_mesh = cloud.generate_surface_mesh(
            self._geometry,
            params=params if params else self._params,
            draft_name=draft_name,
            async_mode=False
        )

    def run_volume_mesher(self, params: SimulationParams = None, draft_name: str = "Volume Mesh"):
        """Run volume mesher with the provided params or defaults"""
        self._check_initialized()

        if self.info.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError("Volume mesher can only be ran by projects with a geometry root asset")
        if self._surface_mesh is None:
            raise Flow360ValueError("This project object is missing a surface mesh, fix by running run_surface_mesher")
        # We should remove usages of cloud static functions here
        self._volume_mesh = cloud.generate_volume_mesh(
            self._surface_mesh,
            params=params if params else self._params,
            draft_name=draft_name,
            async_mode=False
        )

    def run_case(self, params: SimulationParams = None, draft_name: str = "Case"):
        """Run project with the provided params"""
        self._check_initialized()

        if self._volume_mesh is None:
            raise Flow360ValueError("This project object is missing a volume mesh, fix by running run_volume_mesher")
        # We should remove usages of cloud static functions here
        self._case = cloud.generate_volume_mesh(
            self._volume_mesh,
            params=params if params else self._params,
            draft_name=draft_name,
            async_mode=False
        )
