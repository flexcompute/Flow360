"""Project interface for setting up and running simulations"""

# pylint: disable=no-member, too-many-lines
# To be honest I do not know why pylint is insistent on treating
# ProjectMeta instances as FieldInfo, I'd rather not have this line
from __future__ import annotations

import json
from enum import Enum
from typing import Iterable, List, Literal, Optional, Union

import pydantic as pd
from PrettyPrint import PrettyPrintTree
from pydantic import PositiveInt

from flow360.cloud.flow360_requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.geometry import Geometry
from flow360.component.interfaces import (
    GeometryInterface,
    ProjectInterface,
    SurfaceMeshInterfaceV2,
    VolumeMeshInterfaceV2,
)
from flow360.component.project_utils import (
    GeometryFiles,
    SurfaceMeshFile,
    VolumeMeshFile,
    formatting_validation_errors,
    set_up_params_for_uploading,
    show_projects_with_keyword_filter,
    validate_params_with_context,
)
from flow360.component.resource_base import Flow360Resource
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import Draft
from flow360.component.surface_mesh_v2 import SurfaceMeshV2
from flow360.component.utils import (
    AssetShortID,
    get_short_asset_id,
    parse_datetime,
    wrapstring,
)
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.exceptions import Flow360FileError, Flow360ValueError, Flow360WebError
from flow360.log import log
from flow360.version import __solver_version__

AssetOrResource = Union[type[AssetBase], type[Flow360Resource]]


class RootType(Enum):
    """
    Enum for root object types in the project.

    Attributes
    ----------
    GEOMETRY : str
        Represents a geometry root object.
    SURFACE_MESH : str
        Represents a surface mesh root object.
    VOLUME_MESH : str
        Represents a volume mesh root object.
    """

    GEOMETRY = "Geometry"
    SURFACE_MESH = "SurfaceMesh"
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
        Type of the root item (Geometry or SurfaceMesh or VolumeMesh).
    """

    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    name: str = pd.Field()
    root_item_id: str = pd.Field(alias="rootItemId")
    root_item_type: RootType = pd.Field(alias="rootItemType")


class ProjectTreeNode(pd.BaseModel):
    """
    ProjectTreeNode class containing the info of an asset item in a project tree.

    Attributes
    ----------
    asset_id : str
        ID of the asset.
    asset_name : str
        Name of the asset.
    asset_type : str
        Type of the asset.
    parent_id : Optional[str]
        ID of the parent asset.
    case_mesh_id : Optional[str]
        ID of the case's mesh.
    case_mesh_label : Optional[str]
        Label the mesh of a forked case using a different mesh.
    children : List
        List of the child assets of the current asset.
    min_length_short_id : int
        The minimum length of the short asset id, excluding
        hyphen and asset prefix.
    """

    asset_id: str = pd.Field()
    asset_name: str = pd.Field()
    asset_type: str = pd.Field()
    parent_id: Optional[str] = pd.Field(None)
    case_mesh_id: Optional[str] = pd.Field(None)
    case_mesh_label: Optional[str] = pd.Field(None)
    children: List = pd.Field([])
    min_length_short_id: PositiveInt = pd.Field(7)

    def construct_string(self, line_width):
        """Define the output info within when printing a project tree in the terminal"""
        title_line = "<<" + self.asset_type + ">>"
        name_line = f"name: {self.asset_name}"
        id_line = f"id:   {self.short_id}"

        # Dynamically compute the line_width for each asset block to ensure
        # 1. The asset type title always occupies a single line
        # 2. The id and name line width is no more than the input line_width but is as small as possible.
        max_line_width = min(line_width, max(len(name_line), len(id_line)))
        block_line_width = max(len(title_line), max_line_width)

        name_line = wrapstring(long_str=f"name: {self.asset_name}", str_length=block_line_width)
        id_line = wrapstring(long_str=f"id:   {self.short_id}", str_length=block_line_width)
        return f"{title_line.center(block_line_width)}\n{name_line}\n{id_line}"

    def add_child(self, child: ProjectTreeNode):
        """Add a child asset of the current asset"""
        self.children.append(child)

    def remove_child(self, child_to_remove: ProjectTreeNode):
        """Remove a child asset of the current asset"""
        self.children = [child for child in self.children if child is not child_to_remove]

    @property
    def short_id(self) -> str:
        """Compute short asset id"""
        return get_short_asset_id(
            full_asset_id=self.asset_id, num_character=self.min_length_short_id
        )

    @property
    def edge_label(self) -> str:
        """
        Add edge label in the printed project tree to
        display the different volume mesh used in a forked case.
        """
        if self.case_mesh_label:
            prefix = "Using VolumeMesh:\n"
            mesh_short_id = get_short_asset_id(
                full_asset_id=self.case_mesh_label,
                num_character=self.min_length_short_id,
            )
            return prefix + mesh_short_id.center(len(prefix))
        return None


class ProjectTree(pd.BaseModel):
    """
    ProjectTree class containing the project tree.

    Attributes
    ----------
    root : ProjectTreeNode
        Root item of the project.
    nodes : dict[str, ProjectTreeNode]
        Dict of all nodes in the project tree.
    short_id_map: dict[str, List[str]]
        Dict of short_id to full_id mapping, used to ensure every short_id is unique in the project.
    """

    root: ProjectTreeNode = pd.Field(None)
    nodes: dict[str, ProjectTreeNode] = pd.Field({})
    short_id_map: dict[str, List[str]] = pd.Field({})

    def _update_case_mesh_label(self):
        """Check and remove unnecessary case mesh label"""
        for node_id in self._get_asset_ids_by_type(asset_type="Case"):
            node = self.nodes.get(node_id)
            parent_node = self._get_parent_node(node=node)
            if not parent_node:
                continue
            if parent_node.asset_type != "Case" or node.case_mesh_id == parent_node.case_mesh_id:
                node.case_mesh_label = None

    def _update_node_short_id(self):
        """Update the minimum length of short ID to ensure each node has a unique short ID"""
        if len(self.nodes) == len(self.short_id_map):
            pass
        full_id_to_update = []
        short_id_duplicate = []
        for short_id, full_ids in self.short_id_map.items():
            if len(full_ids) > 1:
                short_id_duplicate.append(short_id)
                common_prefix = full_ids[0]
                for full_id in full_ids[1:]:
                    while not full_id.startswith(common_prefix):
                        common_prefix = common_prefix[:-1]
                common_prefix_processed = "".join(common_prefix.split("-")[1:])
                for full_id in full_ids:
                    # pylint: disable=unsubscriptable-object
                    self.nodes[full_id].min_length_short_id = len(common_prefix_processed) + 1
                    full_id_to_update.append(full_id)
        for full_id in full_id_to_update:
            # pylint: disable=unsubscriptable-object
            self.short_id_map.update({self.nodes[full_id].short_id: [full_id]})
        for short_id in short_id_duplicate:
            self.short_id_map.pop(short_id, None)

    def _get_parent_node(self, node: ProjectTreeNode):
        """Get the parent node of the input node"""
        if not node.parent_id:
            return None
        return self.nodes.get(node.parent_id, None)

    def _has_node(self, asset_id: str) -> bool:
        """Use asset_id to check if the asset already exists in the project tree"""
        if asset_id in self.nodes.keys():
            return True
        return False

    def _get_asset_ids_by_type(
        self, asset_type: str = Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"]
    ):
        """Get the list of asset_ids in the project tree by asset_type."""
        return [node.asset_id for node in self.nodes.values() if node.asset_type == asset_type]

    @classmethod
    def _create_new_node(cls, asset_record: dict):
        """Create a new node based on the asset record from API call"""
        parent_id = (
            asset_record["parentCaseId"]
            if asset_record["parentCaseId"]
            else asset_record["parentId"]
        )
        case_mesh_id = asset_record["parentId"] if asset_record["type"] == "Case" else None

        new_node = ProjectTreeNode(
            asset_id=asset_record["id"],
            asset_name=asset_record["name"],
            asset_type=asset_record["type"],
            parent_id=parent_id,
            case_mesh_id=case_mesh_id,
            case_mesh_label=case_mesh_id,
        )

        return new_node

    def _update_short_id_map(self, new_node: ProjectTreeNode):
        # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
        if new_node.short_id not in self.short_id_map.keys():
            self.short_id_map[new_node.short_id] = []
        self.short_id_map[new_node.short_id].append(new_node.asset_id)

    def add(self, asset_record: dict):
        """Add new node to the existing project tree"""
        if self._has_node(asset_id=asset_record["id"]):
            return False

        new_node = ProjectTree._create_new_node(asset_record)
        self._update_short_id_map(new_node)
        if new_node.parent_id is None:
            self.root = new_node
        for node in self.nodes.values():
            if node.parent_id == new_node.asset_id:
                new_node.add_child(child=node)
            if node.asset_id == new_node.parent_id:
                node.add_child(child=new_node)
        self.nodes.update({new_node.asset_id: new_node})
        self._update_node_short_id()
        self._update_case_mesh_label()
        return True

    def remove_node(self, node_id: str):
        """Remove node from the tree"""
        node = self.nodes.get(node_id)
        if not node:
            return
        if node.parent_id and self._has_node(node.parent_id):
            # pylint: disable=unsubscriptable-object
            self.nodes[node.parent_id].remove_child(node)
        self.nodes.pop(node.asset_id)

    def construct_tree(self, asset_records: List[dict]):
        """Construct the entire project tree"""
        for asset_record in asset_records:
            new_node = ProjectTree._create_new_node(asset_record)
            self._update_short_id_map(new_node)
            if new_node.parent_id is None:
                self.root = new_node
            self.nodes.update({new_node.asset_id: new_node})

        for node in self.nodes.values():
            if node.parent_id and self._has_node(node.parent_id):
                # pylint: disable=unsubscriptable-object
                self.nodes[node.parent_id].add_child(node)
        self._update_node_short_id()
        self._update_case_mesh_label()

    @pd.validate_call
    def get_full_asset_id(self, query_asset: AssetShortID) -> str:
        """
        Returns full asset id of a certain asset type given the query_id.

        Raises
        ------
        Flow360ValueError
            1. If derived asset type from query_id does not match the asset type.
            2. If query_id is too short.
            3. If query_id is does not exist in the project tree.

        Returns
        -------
        The full asset id.
        """

        asset_type_ids = self._get_asset_ids_by_type(asset_type=query_asset.asset_type)
        if len(asset_type_ids) == 0:
            raise Flow360ValueError(f"No {query_asset.asset_type} is available in this project.")

        if query_asset.asset_id is None:
            # The latest asset of this asset_type will be returned.
            return asset_type_ids[-1]

        for asset_id in asset_type_ids:
            if asset_id.startswith(query_asset.asset_id):
                return asset_id
        raise Flow360ValueError(
            f"This asset does not exist in this project. Please check the input asset ID ({query_asset.asset_id})."
        )


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
    project_tree: ProjectTree = pd.Field()
    solver_version: str = pd.Field(frozen=True)

    _root_asset: Union[Geometry, SurfaceMeshV2, VolumeMeshV2] = pd.PrivateAttr(None)

    _root_webapi: Optional[RestApi] = pd.PrivateAttr(None)
    _project_webapi: Optional[RestApi] = pd.PrivateAttr(None)
    _root_simulation_json: Optional[dict] = pd.PrivateAttr(None)

    @classmethod
    def show_remote(cls, search_keyword: Union[None, str] = None):
        """
        Shows all projects on the cloud.

        Parameters
        ----------
        search_keyword : str, optional

        """
        show_projects_with_keyword_filter(search_keyword)

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
    def length_unit(self) -> LengthType.Positive:
        """
        Returns the length unit of the project.

        Returns
        -------
        LengthType.Positive
            The length unit.
        """

        defaults = self._root_simulation_json

        cache_key = "private_attribute_asset_cache"
        length_key = "project_length_unit"

        if cache_key not in defaults or length_key not in defaults[cache_key]:
            raise Flow360ValueError("[Internal] Simulation params do not contain length unit info.")

        return LengthType.validate(defaults[cache_key][length_key])

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

    def get_surface_mesh(self, asset_id: str = None) -> SurfaceMeshV2:
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
        SurfaceMeshV2
            The surface mesh asset.
        """
        self._check_initialized()
        asset_id = self.project_tree.get_full_asset_id(
            query_asset=AssetShortID(asset_id=asset_id, asset_type="SurfaceMesh")
        )
        return SurfaceMeshV2.from_cloud(id=asset_id)

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
        SurfaceMeshV2
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
        asset_id = self.project_tree.get_full_asset_id(
            query_asset=AssetShortID(asset_id=asset_id, asset_type="VolumeMesh")
        )
        return VolumeMeshV2.from_cloud(id=asset_id)

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
        asset_id = self.project_tree.get_full_asset_id(
            query_asset=AssetShortID(asset_id=asset_id, asset_type="Case")
        )
        return Case.from_cloud(case_id=asset_id)

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

    def get_surface_mesh_ids(self) -> Iterable[str]:
        """
        Returns the available IDs of surface meshes in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        # pylint: disable=protected-access
        return self.project_tree._get_asset_ids_by_type(asset_type="SurfaceMesh")

    def get_volume_mesh_ids(self):
        """
        Returns the available IDs of volume meshes in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        # pylint: disable=protected-access
        return self.project_tree._get_asset_ids_by_type(asset_type="VolumeMesh")

    def get_case_ids(self):
        """
        Returns the available IDs of cases in the project

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        # pylint: disable=protected-access
        return self.project_tree._get_asset_ids_by_type(asset_type="Case")

    @classmethod
    def _detect_asset_type_from_file(
        cls, files
    ) -> Union[GeometryFiles, SurfaceMeshFile, VolumeMeshFile, None]:
        """
        Detects the asset type of a file based on its name or pattern.

        Parameters
        ----------
        file : str or list of str
            The file name or path.

        Returns
        -------
        RootType
            The detected file type.

        Raises
        ------
        Flow360FileError
        """
        validated_objects = []
        errors = [None, None, None]

        for model in [GeometryFiles, SurfaceMeshFile, VolumeMeshFile]:
            try:
                validated_objects.append(model(value=files))
            except pd.ValidationError as e:
                validated_objects.append(None)
                errors.append(e)

        if validated_objects == [None, None, None]:
            raise Flow360FileError(
                f"The given file/s: {files} cannot be recognized as"
                "geometry or surface mesh or volume mesh file."
                f"\nGeometry file error: {errors[0]}"
                f"\nSurfaceMesh file error: {errors[1]}"
                f"\nVolumeMesh file error: {errors[2]}"
            )

        # Checking if the file is both a volume mesh and a surface mesh file:
        if validated_objects[1] and validated_objects[2]:
            raise Flow360FileError(
                f"The given file: {files} may be recognized as both volume mesh and surface mesh input."
                f" Please use `SurfaceMeshFile('{files}')` or `VolumeMeshFile('{files}')` to be specific."
            )
        if sum(item is not None for item in validated_objects) > 1:
            raise Flow360FileError(
                f"[Internal error]: More than one file type recognized ({files})."
            )

        return next((item for item in validated_objects if item is not None), None)

    # pylint: disable=too-many-arguments
    @classmethod
    @pd.validate_call
    def from_file(
        cls,
        *,
        files: Union[GeometryFiles, SurfaceMeshFile, VolumeMeshFile, str, list[str]],
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
        if isinstance(files, (GeometryFiles, SurfaceMeshFile, VolumeMeshFile)):
            validated_files = files
        else:
            validated_files = Project._detect_asset_type_from_file(files)

        if isinstance(validated_files, GeometryFiles):
            draft = Geometry.from_file(
                validated_files.value, name, solver_version, length_unit, tags
            )
        elif isinstance(validated_files, SurfaceMeshFile):
            draft = SurfaceMeshV2.from_file(
                validated_files.value, name, solver_version, length_unit, tags
            )
        elif isinstance(validated_files, VolumeMeshFile):
            draft = VolumeMeshV2.from_file(
                validated_files.value, name, solver_version, length_unit, tags
            )
        else:
            raise Flow360FileError(
                "Cannot detect the intended project root with the given file(s)."
            )

        root_asset = draft.submit()

        if not root_asset:
            raise Flow360ValueError(f"Couldn't initialize asset from {validated_files.value}")
        project_id = root_asset.project_id
        project_api = RestApi(ProjectInterface.endpoint, id=project_id)
        info = project_api.get()
        if not info:
            raise Flow360ValueError(f"Couldn't retrieve project info for {project_id}")
        project = Project(
            metadata=ProjectMeta(**info),
            project_tree=ProjectTree(),
            solver_version=root_asset.solver_version,
        )
        project._project_webapi = project_api
        if isinstance(validated_files, GeometryFiles):
            project._root_webapi = RestApi(GeometryInterface.endpoint, id=root_asset.id)
        elif isinstance(validated_files, SurfaceMeshFile):
            project._root_webapi = RestApi(SurfaceMeshInterfaceV2.endpoint, id=root_asset.id)
        elif isinstance(validated_files, VolumeMeshFile):
            project._root_webapi = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_asset.id)
        project._root_asset = root_asset
        project._get_root_simulation_json()
        project._get_tree_from_cloud()
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

        project_info = AssetShortID(asset_id=project_id, asset_type="Project")
        project_api = RestApi(ProjectInterface.endpoint, id=project_info.asset_id)
        info = project_api.get()
        if not isinstance(info, dict):
            raise Flow360WebError(
                f"Cannot load project {project_info.asset_id}, missing project data."
            )
        if not info:
            raise Flow360WebError(f"Couldn't retrieve project info for {project_info.asset_id}")
        meta = ProjectMeta(**info)
        root_asset = None
        root_type = meta.root_item_type
        if root_type == RootType.GEOMETRY:
            root_asset = Geometry.from_cloud(meta.root_item_id)
        elif root_type == RootType.SURFACE_MESH:
            root_asset = SurfaceMeshV2.from_cloud(meta.root_item_id)
        elif root_type == RootType.VOLUME_MESH:
            root_asset = VolumeMeshV2.from_cloud(meta.root_item_id)
        if not root_asset:
            raise Flow360ValueError(f"Couldn't retrieve root asset for {project_info.asset_id}")
        project = Project(
            metadata=meta, project_tree=ProjectTree(), solver_version=root_asset.solver_version
        )
        project._project_webapi = project_api
        if root_type == RootType.GEOMETRY:
            project._root_asset = root_asset
            project._root_webapi = RestApi(GeometryInterface.endpoint, id=root_asset.id)
        elif root_type == RootType.SURFACE_MESH:
            project._root_asset = root_asset
            project._root_webapi = RestApi(SurfaceMeshInterfaceV2.endpoint, id=root_asset.id)
        elif root_type == RootType.VOLUME_MESH:
            project._root_asset = root_asset
            project._root_webapi = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_asset.id)
        project._get_root_simulation_json()
        project._get_tree_from_cloud()
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

    # pylint: disable=protected-access
    def _get_tree_from_cloud(self, destination_obj: AssetOrResource = None):
        """
        Get the project tree from cloud.

        Parameters
        ----------
        destination_obj : AssetOrResource
            The destination asset after submitting a job. If provided, only assets along
            the path to this asset will be fetched to update the local project tree. Otherwise,
            the entire project tree will be refreshed using the latest tree from cloud.
        """

        self._check_initialized()
        asset_records = []
        if destination_obj:
            method = "path"
            resp = self._project_webapi.get(
                method=method,
                params={
                    "itemId": destination_obj.id,
                    "itemType": destination_obj._cloud_resource_type_name,
                },
            )
            for key, val in resp.items():
                if not val:
                    continue
                if key == "cases":
                    asset_records += resp["cases"]
                    continue
                asset_records.append(val)
        else:
            method = "tree"
            resp = self._project_webapi.get(method=method)
            asset_records = resp["records"]
            self.project_tree = ProjectTree()

        asset_records = sorted(
            asset_records,
            key=lambda d: parse_datetime(d["updatedAt"]),
        )

        if method == "tree":
            self.project_tree.construct_tree(asset_records=asset_records)
            return False

        is_duplicate = True
        for record in asset_records:
            success = self.project_tree.add(asset_record=record)
            if success:
                is_duplicate = False

        return is_duplicate

    def refresh_project_tree(self):
        """Refresh the local project tree by fetching the latest project tree from cloud."""
        return self._get_tree_from_cloud()

    def print_project_tree(self, line_width: int = 30, is_horizontal: bool = True):
        """Print the project tree to the terminal.

        Parameters
        ----------
        line_width : str
            The maximum number of characters in each line.

        is_horizontal : bool
            Choose if the project tree is printed in horizontal (default) or vertical direction.

        """

        PrettyPrintTree(
            get_children=lambda x: x.children,
            get_val=lambda x: x.construct_string(line_width=line_width),
            get_label=lambda x: x.edge_label if x.edge_label else None,
            color="",
            border=True,
            orientation=PrettyPrintTree.Horizontal if is_horizontal else PrettyPrintTree.Vertical,
        )(
            self.project_tree.root,
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
        **kwargs,
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

        params = set_up_params_for_uploading(
            params=params, root_asset=self._root_asset, length_unit=self.length_unit
        )

        params, errors = validate_params_with_context(
            params=params,
            root_item_type=self.metadata.root_item_type.value,
            up_to=target._cloud_resource_type_name,
        )

        if errors is not None:
            error_msg = formatting_validation_errors(errors=errors)
            raise ValueError(
                "\n>> Validation error found in the simulation params! Errors are: " + error_msg
            )

        source_item_type = self.metadata.root_item_type.value if fork_from is None else "Case"
        start_from = kwargs.get("start_from", None)

        draft = Draft.create(
            name=draft_name,
            project_id=self.metadata.id,
            source_item_id=self.metadata.root_item_id if fork_from is None else fork_from.id,
            source_item_type=source_item_type,
            solver_version=solver_version if solver_version else self.solver_version,
            fork_case=fork_from is not None,
        ).submit()

        draft.update_simulation_params(params)

        try:
            destination_id = draft.run_up_to_target_asset(
                target,
                source_item_type=source_item_type,
                use_beta_mesher=use_beta_mesher,
                start_from=start_from,
            )
        except RuntimeError:
            return None

        self._project_webapi.patch(
            # pylint: disable=protected-access
            json={
                "lastOpenItemId": destination_id,
                "lastOpenItemType": target._cloud_resource_type_name,
            }
        )

        if target is SurfaceMeshV2 or target is VolumeMeshV2:
            # Intermediate asset and we should enforce it to contain the entity info from root item.
            # pylint: disable=protected-access
            destination_obj = target.from_cloud(
                destination_id, root_item_entity_info_type=self._root_asset._entity_info_class
            )
        else:
            destination_obj = target.from_cloud(destination_id)

        if not run_async:
            destination_obj.wait()

        is_duplicate = self._get_tree_from_cloud(destination_obj=destination_obj)

        if is_duplicate:
            target_asset_type = target._cloud_resource_type_name
            log.warning(
                f"The {target_asset_type} that matches the input already exists in project. "
                f"No new {target_asset_type} will be generated."
            )
        return destination_obj

    @pd.validate_call
    def generate_surface_mesh(
        self,
        params: SimulationParams,
        name: str = "SurfaceMesh",
        run_async: bool = True,
        solver_version: str = None,
        use_beta_mesher: bool = False,
        **kwargs,
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
        surface_mesh = self._run(
            params=params,
            target=SurfaceMeshV2,
            draft_name=name,
            run_async=run_async,
            fork_from=None,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            **kwargs,
        )
        return surface_mesh

    @pd.validate_call
    def generate_volume_mesh(
        self,
        params: SimulationParams,
        name: str = "VolumeMesh",
        run_async: bool = True,
        solver_version: str = None,
        use_beta_mesher: bool = False,
        **kwargs,
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
        if (
            self.metadata.root_item_type is not RootType.GEOMETRY
            and self.metadata.root_item_type is not RootType.SURFACE_MESH
        ):
            raise Flow360ValueError(
                "Volume mesher can only be run by projects with a geometry or surface mesh root asset"
            )
        volume_mesh = self._run(
            params=params,
            target=VolumeMeshV2,
            draft_name=name,
            run_async=run_async,
            fork_from=None,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            **kwargs,
        )
        return volume_mesh

    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def run_case(
        self,
        params: SimulationParams,
        name: str = "Case",
        run_async: bool = True,
        fork_from: Case = None,
        solver_version: str = None,
        use_beta_mesher: bool = False,
        **kwargs,
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
        case = self._run(
            params=params,
            target=Case,
            draft_name=name,
            run_async=run_async,
            fork_from=fork_from,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            **kwargs,
        )
        return case
