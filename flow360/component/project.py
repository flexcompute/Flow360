"""Project interface for setting up and running simulations"""

# pylint: disable=no-member
# To be honest I do not know why pylint is insistent on treating
# ProjectMeta instances as FieldInfo, I'd rather not have this line
import json
from enum import Enum
from typing import Iterable, List, Optional, Union

import pydantic as pd

from flow360.cloud.flow360_requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.geometry import Geometry
from flow360.component.interfaces import (
    GeometryInterface,
    ProjectInterface,
    VolumeMeshInterfaceV2,
)
from flow360.component.resource_base import Flow360Resource
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    Slice,
)
from flow360.component.simulation.primitives import Box, Cylinder, Edge, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import Draft
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    MeshNameParser,
    ProjectAssetCache,
    match_file_pattern,
)
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.exceptions import Flow360FileError, Flow360ValueError, Flow360WebError
from flow360.version import __solver_version__

AssetOrResource = Union[type[AssetBase], type[Flow360Resource]]
RootAsset = Union[Geometry, VolumeMeshV2]


def _set_up_param_entity_info(entity_info, params: SimulationParams):
    """
    Setting up the entity info part of the params.
    1. For non-persistent entities (AKA draft entities), add the ones used in params.
    2. Add the face/edge tags either by looking at the params' value or deduct the tags according to what is used.
    """

    def _get_tag(entity_registry, entity_type: Union[type[Surface], type[Edge]]):
        group_tag = None
        if not entity_registry.find_by_type(entity_type):
            # Did not use any entity of this type, so we add default grouping tag
            return "edgeId" if entity_type == Edge else "faceId"
        for entity in entity_registry.find_by_type(entity_type):
            if entity.private_attribute_tag_key is None:
                raise Flow360ValueError(
                    f"`{entity_type.__name__}` without tagging information is found."
                    f" Please make sure all `{entity_type.__name__}` come from the geometry and is not created ad-hoc."
                )
            if entity.private_attribute_tag_key == "__standalone__":
                # Does not provide information on what grouping user selected.
                continue
            if group_tag is not None and group_tag != entity.private_attribute_tag_key:
                raise Flow360ValueError(
                    f"Multiple `{entity_type.__name__}` group tags detected in"
                    " the simulation parameters which is not supported."
                )
            group_tag = entity.private_attribute_tag_key
        return group_tag

    entity_registry = params.used_entity_registry
    # Creating draft entities
    for draft_type in [Box, Cylinder, Point, PointArray, Slice]:
        draft_entities = entity_registry.find_by_type(draft_type)
        for draft_entity in draft_entities:
            if draft_entity not in entity_info.draft_entities:
                entity_info.draft_entities.append(draft_entity)

    if isinstance(entity_info, GeometryEntityInfo):
        with model_attribute_unlock(entity_info, "face_group_tag"):
            entity_info.face_group_tag = _get_tag(entity_registry, Surface)
        with model_attribute_unlock(entity_info, "edge_group_tag"):
            entity_info.edge_group_tag = _get_tag(entity_registry, Edge)
    return entity_info


class RootType(Enum):
    """
    Enum for root object types in the project.

    Attributes
    ----------
    GEOMETRY : str
        Represents a geometry root object.
    VOLUME_MESH : str
        Represents a volume mesh root object.
    """

    GEOMETRY = "Geometry"
    VOLUME_MESH = "VolumeMesh"


class ProjectMeta(pd.BaseModel, extra="allow"):
    """
    Metadata class for a project.

    Attributes
    ----------
    user_id : str
        The user ID associated with the project.
    id : str
        The project ID.
    name : str
        The name of the project.
    root_item_id : str
        ID of the root item in the project.
    root_item_type : RootType
        Type of the root item (Geometry or VolumeMesh).
    """

    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    name: str = pd.Field()
    root_item_id: str = pd.Field(alias="rootItemId")
    root_item_type: RootType = pd.Field(alias="rootItemType")


_SurfaceMeshCache = ProjectAssetCache[SurfaceMesh]
_VolumeMeshCache = ProjectAssetCache[VolumeMeshV2]
_CaseCache = ProjectAssetCache[Case]


class Project(pd.BaseModel):
    """
    Project class containing the interface for creating and running simulations.

    Attributes
    ----------
    metadata : ProjectMeta
        Metadata of the project.
    solver_version : str
        Version of the solver being used.
    """

    metadata: ProjectMeta = pd.Field()
    solver_version: str = pd.Field(frozen=True)

    _root_asset: Union[Geometry, VolumeMeshV2] = pd.PrivateAttr(None)

    _volume_mesh_cache: _VolumeMeshCache = pd.PrivateAttr(_VolumeMeshCache())
    _surface_mesh_cache: _SurfaceMeshCache = pd.PrivateAttr(_SurfaceMeshCache())
    _case_cache: _CaseCache = pd.PrivateAttr(_CaseCache())

    _root_webapi: Optional[RestApi] = pd.PrivateAttr(None)
    _project_webapi: Optional[RestApi] = pd.PrivateAttr(None)
    _root_simulation_json: Optional[dict] = pd.PrivateAttr(None)

    @property
    def id(self) -> str:
        """
        Returns the ID of the project.

        Returns
        -------
        str
            The project ID.
        """
        return self.metadata.id

    @property
    def geometry(self) -> Geometry:
        """
        Returns the geometry asset of the project. There is always only one geometry asset per project.

        Raises
        ------
        Flow360ValueError
            If the geometry asset is not available for the project.

        Returns
        -------
        Geometry
            The geometry asset.
        """
        self._check_initialized()
        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Geometry asset is only present in projects initialized from geometry."
            )

        return self._root_asset

    def get_surface_mesh(self, asset_id: str = None) -> SurfaceMesh:
        """
        Returns the surface mesh asset of the project.

        Parameters
        ----------
        asset_id : str, optional
            The ID of the asset from among the generated assets in this project instance. If not provided,
            the property contains the most recently run asset.

        Raises
        ------
        Flow360ValueError
            If the surface mesh asset is not available for the project.

        Returns
        -------
        SurfaceMesh
            The surface mesh asset.
        """
        self._check_initialized()
        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Surface mesh assets are only present in projects initialized from geometry."
            )

        return self._surface_mesh_cache.get_asset(asset_id)

    @property
    def surface_mesh(self):
        """
        Returns the last used surface mesh asset of the project.

        Raises
        ------
        Flow360ValueError
            If the surface mesh asset is not available for the project.

        Returns
        -------
        SurfaceMesh
            The surface mesh asset.
        """
        return self.get_surface_mesh()

    def get_volume_mesh(self, asset_id: str = None) -> VolumeMeshV2:
        """
        Returns the volume mesh asset of the project.

        Parameters
        ----------
        asset_id : str, optional
            The ID of the asset from among the generated assets in this project instance. If not provided,
            the property contains the most recently run asset.

        Raises
        ------
        Flow360ValueError
            If the volume mesh asset is not available for the project.

        Returns
        -------
        VolumeMeshV2
            The volume mesh asset.
        """
        self._check_initialized()
        if self.metadata.root_item_type is RootType.VOLUME_MESH:
            if asset_id is not None:
                raise Flow360ValueError(
                    "Cannot retrieve volume meshes by asset ID in a project created from volume mesh, "
                    "there is only one root volume mesh asset in this project. Use project.volume_mesh()."
                )

            return self._root_asset

        return self._volume_mesh_cache.get_asset(asset_id)

    @property
    def volume_mesh(self):
        """
        Returns the last used volume mesh asset of the project.

        Raises
        ------
        Flow360ValueError
            If the volume mesh asset is not available for the project.

        Returns
        -------
        VolumeMeshV2
            The volume mesh asset.
        """
        return self.get_volume_mesh()

    def get_case(self, asset_id: str = None) -> Case:
        """
        Returns the last used case asset of the project.

        Parameters
        ----------
        asset_id : str, optional
            The ID of the asset from among the generated assets in this project instance. If not provided,
            the property contains the most recently run asset.

        Raises
        ------
        Flow360ValueError
            If the case asset is not available for the project.

        Returns
        -------
        Case
            The case asset.
        """
        self._check_initialized()
        return self._case_cache.get_asset(asset_id)

    @property
    def case(self):
        """
        Returns the case asset of the project.

        Raises
        ------
        Flow360ValueError
            If the case asset is not available for the project.

        Returns
        -------
        Case
            The case asset.
        """
        return self.get_case()

    def get_cached_surface_meshes(self) -> Iterable[str]:
        """
        Returns the available IDs of surface meshes in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Surface mesh assets are only present in objects initialized from geometry."
            )

        return self._surface_mesh_cache.get_ids()

    def get_cached_volume_meshes(self):
        """
        Returns the available IDs of volume meshes in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        if self.metadata.root_item_type is RootType.VOLUME_MESH:
            raise Flow360ValueError(
                "Cannot retrieve volume meshes by asset ID in a project created from volume mesh, "
                "there is only one root volume mesh asset in this project. Use project.volume_mesh()."
            )

        return self._volume_mesh_cache.get_ids()

    def get_cached_cases(self):
        """
        Returns the available IDs of cases in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        return self._case_cache.get_ids()

    @classmethod
    def _detect_asset_type_from_file(cls, file):
        """
        Detects the asset type of a file based on its name or pattern.

        Parameters
        ----------
        file : str
            The file name or path.

        Returns
        -------
        RootType
            The detected root type (Geometry or VolumeMesh).

        Raises
        ------
        Flow360FileError
            If the file does not match any known patterns.
        """
        if match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, file):
            return RootType.GEOMETRY
        try:
            parser = MeshNameParser(file)
            if parser.is_valid_volume_mesh():
                return RootType.VOLUME_MESH
        except Flow360FileError:
            pass

        raise Flow360FileError(
            f"{file} is not a geometry or volume mesh file required for project initialization. "
            "Accepted formats are: "
            f"{SUPPORTED_GEOMETRY_FILE_PATTERNS} (geometry)"
            f"{MeshNameParser.all_patterns(mesh_type='volume')} (volume mesh)"
        )

    # pylint: disable=too-many-arguments
    @classmethod
    @pd.validate_call
    def from_file(
        cls,
        file: str = None,
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        """
        Initializes a project from a file.

        Parameters
        ----------
        file : str
            Path to the file.
        name : str, optional
            Name of the project (default is None).
        solver_version : str, optional
            Version of the solver (default is None).
        length_unit : LengthUnitType, optional
            Unit of length (default is "m").
        tags : list of str, optional
            Tags to assign to the project (default is None).

        Returns
        -------
        Project
            An instance of the project.

        Raises
        ------
        Flow360ValueError
            If the project cannot be initialized from the file.
        """
        root_asset = None
        root_type = cls._detect_asset_type_from_file(file)
        if root_type == RootType.GEOMETRY:
            draft = Geometry.from_file(file, name, solver_version, length_unit, tags)
            root_asset = draft.submit()
        elif root_type == RootType.VOLUME_MESH:
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
        project._project_webapi = project_api
        if root_type == RootType.GEOMETRY:
            project._root_asset = root_asset
            project._root_webapi = RestApi(GeometryInterface.endpoint, id=root_asset.id)
        elif root_type == RootType.VOLUME_MESH:
            project._root_asset = root_asset
            project._root_webapi = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_asset.id)
        project._get_root_simulation_json()
        return project

    @classmethod
    @pd.validate_call
    def from_cloud(cls, project_id: str):
        """
        Loads a project from the cloud.

        Parameters
        ----------
        project_id : str
            ID of the project.

        Returns
        -------
        Project
            An instance of the project.

        Raises
        ------
        Flow360WebError
            If the project cannot be loaded from the cloud.
        Flow360ValueError
            If the root asset cannot be retrieved for the project.
        """
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
        if not root_asset:
            raise Flow360ValueError(f"Couldn't retrieve root asset for {project_id}")
        project = Project(metadata=meta, solver_version=root_asset.solver_version)
        project._project_webapi = project_api
        if root_type == RootType.GEOMETRY:
            project._root_asset = root_asset
            project._root_webapi = RestApi(GeometryInterface.endpoint, id=root_asset.id)
        elif root_type == RootType.VOLUME_MESH:
            project._root_asset = root_asset
            project._root_webapi = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_asset.id)
        project._get_root_simulation_json()
        return project

    def _check_initialized(self):
        """
        Checks if the project instance has been initialized correctly.

        Raises
        ------
        Flow360ValueError
            If the project is not initialized.
        """
        if not self.metadata or not self.solver_version or not self._root_asset:
            raise Flow360ValueError(
                "Project not initialized - use Project.from_file or Project.from_cloud"
            )

    def _get_root_simulation_json(self):
        """
        Loads the default simulation JSON for the project based on the root asset type.

        Raises
        ------
        Flow360ValueError
            If the root item type or ID is missing from project metadata.
        Flow360WebError
            If the simulation JSON cannot be retrieved.
        """
        self._check_initialized()
        root_type = self.metadata.root_item_type
        root_id = self.metadata.root_item_id
        if not root_type or not root_id:
            raise Flow360ValueError("Root item type or ID is missing from project metadata")
        resp = self._root_webapi.get(method="simulation/file", params={"type": "simulation"})
        if not isinstance(resp, dict) or "simulationJson" not in resp:
            raise Flow360WebError("Couldn't retrieve default simulation JSON for the project")
        simulation_json = json.loads(resp["simulationJson"])
        self._root_simulation_json = simulation_json

    # pylint: disable=too-many-arguments, too-many-locals
    def _run(
        self,
        *,
        params: SimulationParams,
        target: AssetOrResource,
        draft_name: str,
        fork_from: Case,
        run_async: bool,
        solver_version: str,
        use_beta_mesher: bool,
    ):
        """
        Runs a simulation for the project.

        Parameters
        ----------
        params : SimulationParams
            The simulation parameters to use for the run
        target : AssetOrResource
            The target asset or resource to run the simulation against.
        draft_name : str, optional
            The name of the draft to create for the simulation run (default is None).
        fork : bool, optional
            Indicates if the simulation should fork the existing case (default is False).
        run_async : bool, optional
            Specifies whether the simulation should run asynchronously (default is True).

        Returns
        -------
        AssetOrResource
            The destination asset

        Raises
        ------
        Flow360ValueError
            If the simulation parameters lack required length unit information, or if the
            root asset (Geometry or VolumeMesh) is not initialized.
        """

        defaults = self._root_simulation_json

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

        root_asset = self._root_asset

        draft = Draft.create(
            name=draft_name,
            project_id=self.metadata.id,
            source_item_id=self.metadata.root_item_id if fork_from is None else fork_from.id,
            source_item_type=(self.metadata.root_item_type.value if fork_from is None else "Case"),
            solver_version=solver_version if solver_version else self.solver_version,
            fork_case=fork_from is not None,
        ).submit()

        # Check if there are any new draft entities that have been added in the params by the user
        entity_info = _set_up_param_entity_info(root_asset.entity_info, params)

        with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
            params.private_attribute_asset_cache.project_entity_info = entity_info

        draft.update_simulation_params(params)

        destination_id = draft.run_up_to_target_asset(target, use_beta_mesher=use_beta_mesher)

        self._project_webapi.patch(
            # pylint: disable=protected-access
            json={
                "lastOpenItemId": destination_id,
                "lastOpenItemType": target._cloud_resource_type_name,
            }
        )

        if target is SurfaceMesh or target is VolumeMeshV2:
            # Intermediate asset and we should enforce it to contain the entity info from root item.
            # pylint: disable=protected-access
            destination_obj = target.from_cloud(
                destination_id, root_item_entity_info_type=self._root_asset._entity_info_class
            )
        else:
            destination_obj = target.from_cloud(destination_id)

        if not run_async:
            destination_obj.wait()

        return destination_obj

    @pd.validate_call
    def generate_surface_mesh(
        self,
        params: SimulationParams,
        name: str = "SurfaceMesh",
        run_async: bool = True,
        solver_version: str = None,
        use_beta_mesher: bool = False,
    ):
        """
        Runs the surface mesher for the project.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters for running the mesher.
        name : str, optional
            Name of the surface mesh (default is "SurfaceMesh").
        run_async : bool, optional
            Whether to run the mesher asynchronously (default is True).
        solver_version : str, optional
            Optional solver version to use during this run (defaults to the project solver version)

        Raises
        ------
        Flow360ValueError
            If the root item type is not Geometry.
        """
        self._check_initialized()
        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Surface mesher can only be run by projects with a geometry root asset"
            )
        self._surface_mesh_cache.add_asset(
            self._run(
                params=params,
                target=SurfaceMesh,
                draft_name=name,
                run_async=run_async,
                fork_from=None,
                solver_version=solver_version,
                use_beta_mesher=use_beta_mesher,
            )
        )

    @pd.validate_call
    def generate_volume_mesh(
        self,
        params: SimulationParams,
        name: str = "VolumeMesh",
        run_async: bool = True,
        solver_version: str = None,
        use_beta_mesher: bool = False,
    ):
        """
        Runs the volume mesher for the project.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters for running the mesher.
        name : str, optional
            Name of the volume mesh (default is "VolumeMesh").
        run_async : bool, optional
            Whether to run the mesher asynchronously (default is True).
        solver_version : str, optional
            Optional solver version to use during this run (defaults to the project solver version)

        Raises
        ------
        Flow360ValueError
            If the root item type is not Geometry.
        """
        self._check_initialized()
        if self.metadata.root_item_type is not RootType.GEOMETRY:
            raise Flow360ValueError(
                "Volume mesher can only be run by projects with a geometry root asset"
            )
        self._volume_mesh_cache.add_asset(
            self._run(
                params=params,
                target=VolumeMeshV2,
                draft_name=name,
                run_async=run_async,
                fork_from=None,
                solver_version=solver_version,
                use_beta_mesher=use_beta_mesher,
            )
        )

    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def run_case(
        self,
        params: SimulationParams,
        name: str = "Case",
        run_async: bool = True,
        fork_from: Case = None,
        solver_version: str = None,
        use_beta_mesher: bool = False,
    ):
        """
        Runs a case for the project.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters for running the case.
        name : str, optional
            Name of the case (default is "Case").
        run_async : bool, optional
            Whether to run the case asynchronously (default is True).
        fork_from : Case, optional
            Which Case we should fork from (if fork).
        solver_version : str, optional
            Optional solver version to use during this run (defaults to the project solver version)
        """
        self._check_initialized()
        self._case_cache.add_asset(
            self._run(
                params=params,
                target=Case,
                draft_name=name,
                run_async=run_async,
                fork_from=fork_from,
                solver_version=solver_version,
                use_beta_mesher=use_beta_mesher,
            )
        )
