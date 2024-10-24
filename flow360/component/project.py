"""Project interface for setting up and running simulations"""

# pylint: disable=no-member
# To be honest I do not know why pylint is insistent on treating
# ProjectMeta instances as FieldInfo, I'd rather not have this line
import json
from enum import Enum
from typing import List, Optional, Union

import pydantic as pd

from flow360 import Case, SurfaceMesh
from flow360.cloud.requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.geometry import Geometry
from flow360.component.interfaces import (
    GeometryInterface,
    ProjectInterface,
    VolumeMeshInterfaceV2,
)
from flow360.component.resource_base import Flow360Resource
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import Draft
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    MeshNameParser,
    match_file_pattern,
)
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.exceptions import Flow360FileError, Flow360ValueError, Flow360WebError

# This is used before all resources are moved to V2 API as AssetBase subclasses
AssetOrResource = Union[type[AssetBase], type[Flow360Resource]]
RootAsset = Union[Geometry, VolumeMeshV2]


class RootType(Enum):
    """
    Type of a root object of the project
    """

    GEOMETRY = "Geometry"
    VOLUME_MESH = "VolumeMesh"
    # SURFACE_MESH = "SurfaceMesh" - supported in a future iteration


class ProjectMeta(pd.BaseModel, extra=pd.Extra.allow):
    """
    Metadata of the project
    """

    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    name: str = pd.Field()
    root_item_id: str = pd.Field(alias="rootItemId")
    root_item_type: RootType = pd.Field(alias="rootItemType")


class Project(pd.BaseModel):
    """
    Project class containing the interface for creating and running simulations
    """

    metadata: ProjectMeta = pd.Field()
    solver_version: str = pd.Field(frozen=True)

    # Right now the cached assets are all related to the
    # root asset - we will have full project tree traversal
    # in a future iteration.
    _geometry: Optional[Geometry] = pd.PrivateAttr(None)
    _volume_mesh: Optional[VolumeMeshV2] = pd.PrivateAttr(None)
    _surface_mesh: Optional[SurfaceMesh] = pd.PrivateAttr(None)
    _case: Optional[Case] = pd.PrivateAttr(None)

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
    def volume_mesh(self) -> VolumeMeshV2:
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

        try:
            parser = MeshNameParser(file)
            if parser.is_valid_volume_mesh():
                return RootType.VOLUME_MESH
        except Flow360FileError:
            pass

        raise Flow360FileError(
            f"{file} is not a geometry or volume mesh file required for project initialization."
        )

    # pylint: disable=too-many-arguments
    @classmethod
    def from_file(
        cls,
        file: str = None,
        name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        """
        Populates project data from a file

        Args:
            name (): Name of the new project
            file ():
            solver_version ():
            length_unit ():
            tags ():

        Returns:

        """

        root_asset = None

        root_type = cls._detect_root_type(file)

        if root_type == RootType.GEOMETRY:
            # Create a draft geometry asset and submit
            draft = Geometry.from_file(file, name, solver_version, length_unit, tags)
            root_asset = draft.submit()
        elif root_type == RootType.VOLUME_MESH:
            # Create a draft volume mesh asset and submit
            draft = VolumeMeshV2.from_file(file, name, solver_version, length_unit, tags)
            root_asset = draft.submit()

        if not root_asset:
            raise Flow360ValueError(f"Couldn't initialize asset from {file}")

        project_id = root_asset.project_id

        project_api = RestApi(ProjectInterface.endpoint, id=project_id)
        info = project_api.get()

        if not info:
            raise Flow360ValueError(f"Couldn't retrieve project info for {project_id}")

        project = Project(metadata=ProjectMeta(**info), solver_version=root_asset.solver_version)

        if root_type == RootType.GEOMETRY:
            project._geometry = root_asset
        elif root_type == RootType.VOLUME_MESH:
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

        meta = ProjectMeta(**info)

        root_asset = None
        root_type = meta.root_item_type

        if root_type == RootType.GEOMETRY:
            root_asset = Geometry.from_cloud(meta.root_item_id)
        elif root_type == RootType.VOLUME_MESH:
            root_asset = VolumeMeshV2.from_cloud(meta.root_item_id)

        project = Project(metadata=meta, solver_version=root_asset.solver_version)

        if not root_asset:
            raise Flow360ValueError(f"Couldn't retrieve root asset for {project_id}")

        if root_type == RootType.GEOMETRY:
            project._geometry = root_asset
        elif root_type == RootType.VOLUME_MESH:
            project._volume_mesh = root_asset

        return project

    def _check_initialized(self):
        """
        Check if the Project instance has been initialized correctly
        """
        if not self.metadata or not self.solver_version:
            raise Flow360ValueError(
                "Project is not initialized - use Project.from_file or Project.from_cloud"
            )

    def get_simulation_json(self) -> dict:
        """
        Get default simulation JSON for the project based on the root asset type
        """
        self._check_initialized()

        root_type = self.metadata.root_item_type
        root_id = self.metadata.root_item_id

        if not root_type or not root_id:
            raise Flow360ValueError("Root item type or ID is missing from project metadata")

        root_api = None

        if root_type == RootType.GEOMETRY:
            root_api = RestApi(GeometryInterface.endpoint, id=root_id)
        elif root_type == RootType.VOLUME_MESH:
            root_api = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_id)

        resp = root_api.get(method="simulation/file", params={"type": "simulation"})

        if not isinstance(resp, dict) or "simulationJson" not in resp:
            raise Flow360WebError("Root item type or ID is missing from project metadata")

        simulation_json = json.loads(resp["simulationJson"])

        return simulation_json

    # pylint: disable=too-many-locals, too-many-arguments
    def _run(
        self,
        params: SimulationParams,
        target: AssetOrResource,
        draft_name: str = None,
        fork: bool = False,
        run_async: bool = True,
    ):
        defaults = self.get_simulation_json()

        cache_key = "private_attribute_asset_cache"
        length_key = "project_length_unit"

        if cache_key not in defaults:
            if length_key not in defaults[cache_key]:
                raise Flow360ValueError("Simulation params do not contain default length unit info")

        length_unit = defaults[cache_key][length_key]

        with model_attribute_unlock(params.private_attribute_asset_cache, length_key):
            params.private_attribute_asset_cache.project_length_unit = LengthType.validate(
                length_unit
            )

        root_asset = None
        root_type = self.metadata.root_item_type

        if root_type == RootType.GEOMETRY:
            root_asset = self._geometry
        elif root_type == RootType.VOLUME_MESH:
            root_asset = self._volume_mesh

        if not root_asset:
            raise Flow360ValueError("Root asset is not initialized")

        draft = Draft.create(
            name=draft_name,
            project_id=self.metadata.id,
            source_item_id=self.metadata.root_item_id,
            source_item_type=root_type.value,
            solver_version=self.solver_version,
            fork_case=fork,
        ).submit()

        entity_info = root_asset.entity_info
        registry = params.used_entity_registry
        old_draft_entities = entity_info.draft_entities
        for _, old_entity in enumerate(old_draft_entities):
            try:
                registry.find_by_naming_pattern(old_entity.name)
            except ValueError:
                continue

        with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
            params.private_attribute_asset_cache.project_entity_info = entity_info

        draft.update_simulation_params(params)
        destination_id = draft.run_up_to_target_asset(target)

        RestApi(ProjectInterface.endpoint, id=self.metadata.id).patch(
            json={
                "lastOpenItemId": destination_id,
                "lastOpenItemType": target.__name__,
            }
        )

        destination_obj = target.from_cloud(destination_id)

        if not run_async:
            destination_obj.wait()

        return destination_obj

    def run_surface_mesher(
        self, params: SimulationParams, draft_name: str = "SurfaceMesh", run_async: bool = True
    ):
        """Run surface mesher with the provided params or defaults"""
        self._check_initialized()

        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Surface mesher can only be ran by projects with a geometry root asset"
            )

        self._surface_mesh = self._run(
            params=params, target=SurfaceMesh, draft_name=draft_name, run_async=run_async
        )

    def run_volume_mesher(
        self, params: SimulationParams, draft_name: str = "VolumeMesh", run_async: bool = True
    ):
        """Run volume mesher with the provided params or defaults"""
        self._check_initialized()

        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Volume mesher can only be ran by projects with a geometry root asset"
            )

        self._volume_mesh = self._run(
            params=params, target=VolumeMeshV2, draft_name=draft_name, run_async=run_async
        )

    def run_case(self, params: SimulationParams, draft_name: str = "Case", run_async: bool = True):
        """Run project with the provided params"""
        self._check_initialized()

        self._case = self._run(
            params=params, target=Case, draft_name=draft_name, run_async=run_async
        )
