"""Project interface for setting up and running simulations"""

# pylint: disable=no-member, too-many-lines
# To be honest I do not know why pylint is insistent on treating
# ProjectMeta instances as FieldInfo, I'd rather not have this line
from __future__ import annotations

import json
from enum import Enum
from typing import Iterable, List, Literal, Optional, Union

import pydantic as pd
import typing_extensions
from PrettyPrint import PrettyPrintTree
from pydantic import PositiveInt

from flow360.cloud.flow360_requests import LengthUnitType, RenameAssetRequestV2
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
    get_project_records,
    set_up_params_for_uploading,
    show_projects_with_keyword_filter,
    validate_params_with_context,
)
from flow360.component.resource_base import Flow360Resource
from flow360.component.simulation.folder import Folder
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import Draft
from flow360.component.surface_mesh_v2 import SurfaceMeshV2
from flow360.component.utils import (
    AssetShortID,
    GeometryFiles,
    SurfaceMeshFile,
    VolumeMeshFile,
    formatting_validation_errors,
    get_short_asset_id,
    parse_datetime,
    wrapstring,
)
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.exceptions import Flow360FileError, Flow360ValueError, Flow360WebError
from flow360.log import log
from flow360.plugins.report.report import get_default_report_summary_template
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
    tags : List[str]
        List of tags associated with the project.
    root_item_id : str
        ID of the root item in the project.
    root_item_type : RootType
        Type of the root item (Geometry or SurfaceMesh or VolumeMesh).
    """

    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    name: str = pd.Field()
    tags: List[str] = pd.Field(default_factory=list)
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


# pylint: disable=too-many-public-methods
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
    def tags(self) -> List[str]:
        """
        Returns the tags of the project.

        Returns
        -------
        List[str]
            List of the project's tags.
        """
        return self.metadata.tags

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
    def surface_mesh(self) -> SurfaceMeshV2:
        """
        Returns the last used surface mesh asset of the project.

        If the project is initialized from surface mesh, the surface mesh asset is the root asset.

        Raises
        ------
        Flow360ValueError
            If the surface mesh asset is not available for the project.

        Returns
        -------
        SurfaceMeshV2
            The surface mesh asset.
        """
        if self.metadata.root_item_type is RootType.SURFACE_MESH:
            return self._root_asset
        log.warning(
            f"Accessing surface mesh from a project initialized from {self.metadata.root_item_type.name}. "
            "Please use the root asset for assigning entities to SimulationParams."
        )
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
    def volume_mesh(self) -> VolumeMeshV2:
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
        if self.metadata.root_item_type is RootType.VOLUME_MESH:
            return self._root_asset
        log.warning(
            f"Accessing volume mesh from a project initialized from {self.metadata.root_item_type.name}. "
            "Please use the root asset for assigning entities to SimulationParams."
        )
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

    def get_case_ids(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        Returns the available IDs of cases in the project, optionally filtered by tags.

        Parameters
        ----------
        tags : List[str], optional
            List of tags to filter cases by. If None or empty tags list, returns all case IDs.

        Returns
        -------
        Iterable[str]
            An iterable of case IDs. If tags are provided, filters to return only
            case IDs that have at least one matching tag.
        """
        # pylint: disable=protected-access
        all_case_ids = self.project_tree._get_asset_ids_by_type(asset_type="Case")

        if not tags:
            return all_case_ids

        # Filter cases by tags
        filtered_case_ids = []
        for case_id in all_case_ids:
            case = self.get_case(asset_id=case_id)
            if set(tags) & set(case.info_v2.tags):
                filtered_case_ids.append(case_id)

        return filtered_case_ids

    @classmethod
    def get_project_ids(cls, tags: Optional[List[str]] = None) -> List[str]:
        """
        Returns the available IDs of projects, optionally filtered by tags.

        Parameters
        ----------
        tags : List[str], optional
            List of tags to filter projects by. If None, returns all project IDs.

        Returns
        -------
        List[str]
            A list of project IDs. If tags are provided, filters to return only
            project IDs that have at least one matching tag.
        """
        project_records, _ = get_project_records("", tags=tags)
        return [record.project_id for record in project_records.records]

    # pylint: disable=too-many-arguments
    @classmethod
    def _create_project_from_files(
        cls,
        *,
        files: Union[GeometryFiles, SurfaceMeshFile, VolumeMeshFile],
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        run_async: bool = False,
        folder: Optional[Folder] = None,
    ):
        """
        Initializes a project from a file.

        Parameters
        ----------
        files : Union[GeometryFiles, SurfaceMeshFile, VolumeMeshFile]
            Path to the files.
        name : str, optional
            Name of the project (default is None).
        solver_version : str, optional
            Version of the solver (default is None).
        length_unit : LengthUnitType, optional
            Unit of length (default is "m").
        tags : list of str, optional
            Tags to assign to the project (default is None).
        run_async : bool, optional
            Whether to create the project asynchronously (default is False).
        folder : Optional[Folder], optional
            Parent folder for the project. If None, creates in root.

        Returns
        -------
        Project
            An instance of the project. Or Project ID when run_async is True.

        Raises
        ------
        Flow360ValueError
            If the project cannot be initialized from the file.
        """
        root_asset = None

        # pylint:disable = protected-access
        files._check_files_existence()

        if isinstance(files, GeometryFiles):
            draft = Geometry.from_file(
                files.file_names, name, solver_version, length_unit, tags, folder=folder
            )
        elif isinstance(files, SurfaceMeshFile):
            draft = SurfaceMeshV2.from_file(
                files.file_names, name, solver_version, length_unit, tags, folder=folder
            )
        elif isinstance(files, VolumeMeshFile):
            draft = VolumeMeshV2.from_file(
                files.file_names, name, solver_version, length_unit, tags, folder=folder
            )
        else:
            raise Flow360FileError(
                "Cannot detect the intended project root with the given file(s)."
            )

        root_asset = draft.submit(run_async=run_async)
        if run_async:
            log.info(
                f"The input file(s) has been successfully uploaded to project: {root_asset.project_id} "
                "and is being processed on cloud. Only the project ID string is returned. "
                "To retrieve this project later, use 'Project.from_cloud(project_id)'. "
            )
            return root_asset.project_id

        if not root_asset:
            raise Flow360ValueError(f"Couldn't initialize asset from {files.file_names}")
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
        if isinstance(files, GeometryFiles):
            project._root_webapi = RestApi(GeometryInterface.endpoint, id=root_asset.id)
        elif isinstance(files, SurfaceMeshFile):
            project._root_webapi = RestApi(SurfaceMeshInterfaceV2.endpoint, id=root_asset.id)
        elif isinstance(files, VolumeMeshFile):
            project._root_webapi = RestApi(VolumeMeshInterfaceV2.endpoint, id=root_asset.id)
        project._root_asset = root_asset
        project._get_root_simulation_json()
        project._get_tree_from_cloud()
        return project

    @classmethod
    @pd.validate_call(
        config={
            "arbitrary_types_allowed": True
        }  # Folder (v2: component/simulation/folder.py) does not have validate() defined
    )
    def from_geometry(
        cls,
        files: Union[str, list[str]],
        /,
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        run_async: bool = False,
        folder: Optional[Folder] = None,
    ):
        """
        Initializes a project from local geometry files.

        Parameters
        ----------
        files : Union[str, list[str]] (positional argument only)
            Geometry file paths.
        name : str, optional
            Name of the project (default is None).
        solver_version : str, optional
            Version of the solver (default is None).
        length_unit : LengthUnitType, optional
            Unit of length (default is "m").
        tags : list of str, optional
            Tags to assign to the project (default is None).
        run_async : bool, optional
            Whether to create project asynchronously (default is False).
        folder : Optional[Folder], optional
            Parent folder for the project. If None, creates in root.

        Returns
        -------
        Project
            An instance of the project. Or Project ID when run_async is True.

        Raises
        ------
        Flow360FileError
            If the project cannot be initialized from the file.

        Example
        -------
        >>> my_project = fl.Project.from_geometry(
        ...     "/path/to/my/geometry/my_geometry.csm",
        ...     name="My_Project_name",
        ...     solver_version="release-Major.Minor"
        ...     length_unit="cm"
        ...     tags=["Quarter 1", "Revision 2"]
        ... )
        ====
        """
        try:
            validated_files = GeometryFiles(file_names=files)
        except pd.ValidationError as err:
            # pylint:disable = raise-missing-from
            raise Flow360FileError(f"Geometry file error: {str(err)}")

        return cls._create_project_from_files(
            files=validated_files,
            name=name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
            run_async=run_async,
            folder=folder,
        )

    @classmethod
    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def from_surface_mesh(
        cls,
        file: str,
        /,
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        run_async: bool = False,
        folder: Optional[Folder] = None,
    ):
        """
        Initializes a project from a local surface mesh file.

        Parameters
        ----------
        file : str (positional argument only)
            Surface mesh file path. For UGRID file the mapbc
            file needs to be renamed with the same prefix under same folder.
        name : str, optional
            Name of the project (default is None).
        solver_version : str, optional
            Version of the solver (default is None).
        length_unit : LengthUnitType, optional
            Unit of length (default is "m").
        tags : list of str, optional
            Tags to assign to the project (default is None).
        run_async : bool, optional
            Whether to create project asynchronously (default is False).
        folder : Optional[Folder], optional
            Parent folder for the project. If None, creates in root.

        Returns
        -------
        Project
            An instance of the project. Or Project ID when run_async is True.

        Raises
        ------
        Flow360FileError
            If the project cannot be initialized from the file.

        Example
        -------
        >>> my_project = fl.Project.from_surface_mesh(
        ...     "/path/to/my/mesh/my_mesh.ugrid",
        ...     name="My_Project_name",
        ...     solver_version="release-Major.Minor"
        ...     length_unit="inch"
        ...     tags=["Quarter 1", "Revision 2"]
        ... )
        ====
        """

        try:
            validated_files = SurfaceMeshFile(file_names=file)
        except pd.ValidationError as err:
            # pylint:disable = raise-missing-from
            raise Flow360FileError(f"Surface mesh file error: {str(err)}")

        return cls._create_project_from_files(
            files=validated_files,
            name=name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
            run_async=run_async,
            folder=folder,
        )

    @classmethod
    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def from_volume_mesh(
        cls,
        file: str,
        /,
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        run_async: bool = False,
        folder: Optional[Folder] = None,
    ):
        """
        Initializes a project from a local volume mesh file.

        Parameters
        ----------
        file : str (positional argument only)
            Volume mesh file path. For UGRID file the mapbc
            file needs to be renamed with the same prefix under same folder.
        name : str, optional
            Name of the project (default is None).
        solver_version : str, optional
            Version of the solver (default is None).
        length_unit : LengthUnitType, optional
            Unit of length (default is "m").
        tags : list of str, optional
            Tags to assign to the project (default is None).
        run_async : bool, optional
            Whether to create project asynchronously (default is False).
        folder : Optional[Folder], optional
            Parent folder for the project. If None, creates in root.

        Returns
        -------
        Project
            An instance of the project.

        Raises
        ------
        Flow360FileError
            If the project cannot be initialized from the file. Or Project ID when run_async is True.

        Example
        -------
        >>> my_project = fl.Project.from_volume_mesh(
        ...     "/path/to/my/mesh/my_mesh.cgns",
        ...     name="My_Project_name",
        ...     solver_version="release-Major.Minor"
        ...     length_unit="inch"
        ...     tags=["Quarter 1", "Revision 2"]
        ... )
        ====
        """

        try:
            validated_files = VolumeMeshFile(file_names=file)
        except pd.ValidationError as err:
            # pylint:disable = raise-missing-from
            raise Flow360FileError(f"Volume mesh file error: {str(err)}")

        return cls._create_project_from_files(
            files=validated_files,
            name=name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
            run_async=run_async,
            folder=folder,
        )

    @classmethod
    @typing_extensions.deprecated(
        "Creating project with `from_file` is deprecated. Please use `from_geometry()`, "
        "`from_surface_mesh()` or `from_volume_mesh()` instead.",
        category=None,
    )
    @pd.validate_call
    def from_file(
        cls,
        file: Union[str, list[str]],
        name: str = None,
        solver_version: str = __solver_version__,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
        run_async: bool = False,
    ):
        """
        [Deprecated function]
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
        run_async : bool, optional
            Whether to create project asynchronously (default is False).

        Returns
        -------
        Project
            An instance of the project. Or Project ID when run_async is True.

        Raises
        ------
        Flow360ValueError
            If the project cannot be initialized from the file.
        """

        log.warning(
            "DeprecationWarning: Creating project with `from_file` is deprecated. "
            + "Please use `from_geometry()`, `from_surface_mesh()` "
            + "or `from_volume_mesh()` instead."
        )

        def _detect_input_file_type(file: Union[str, list[str]]):
            errors = []
            for model in [GeometryFiles, VolumeMeshFile]:
                try:
                    return model(file_names=file)
                except pd.ValidationError as e:
                    errors.append(e)
            raise Flow360FileError(f"Input file {file} cannot be recognized.\nErrors: {errors}")

        return cls._create_project_from_files(
            files=_detect_input_file_type(file=file),
            name=name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
            run_async=run_async,
        )

    @classmethod
    def _get_user_requested_entity_info(
        cls,
        *,
        current_project_id: str,
        new_run_from: Optional[Union[Geometry, SurfaceMeshV2, VolumeMeshV2, Case]] = None,
    ):
        """
        Get the entity info requested by the users when they specify `new_run_from` when calling
        Project.from_cloud()
        """

        user_requested_entity_info = None
        if new_run_from is None:
            return user_requested_entity_info

        if new_run_from.project_id is None:
            # Can only happen to case created using V1 interface.
            raise ValueError(
                "The supplied case resource for `new_run_from` was created using old interface and "
                "cannot be used as the starting point of a new run."
            )
        if current_project_id != new_run_from.project_id:
            raise ValueError(
                "The supplied cloud resource for `new_run_from` does not belong to the project."
            )

        if isinstance(new_run_from, Case):
            user_requested_entity_info = new_run_from.get_simulation_params()
        if isinstance(new_run_from, (Geometry, SurfaceMeshV2, VolumeMeshV2)):
            user_requested_entity_info = new_run_from.params

        return user_requested_entity_info

    @classmethod
    @pd.validate_call(
        config={"arbitrary_types_allowed": True}  # Geometry etc do not have validate() defined
    )
    def from_cloud(
        cls,
        project_id: str,
        *,
        new_run_from: Optional[Union[Geometry, SurfaceMeshV2, VolumeMeshV2, Case]] = None,
    ):
        """
        Loads a project from the cloud.

        Parameters
        ----------
        project_id : str
            ID of the project.
        new_run_from: Optional[Union[Geometry, SurfaceMeshV2, VolumeMeshV2, Case]]
            The cloud resource that the current run should be based on.
            The root asset will use entity settings (grouping, transformation etc) from this resource.
            This results in the same behavior when user clicks New run on webUI.
            By default this will be the root asset (what user uploaded) of the project.

            TODO: We can add 'last' as one option to automatically start
            from the latest created asset within the project.

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

        if (
            new_run_from is not None
            and isinstance(new_run_from, (Geometry, SurfaceMeshV2, VolumeMeshV2, Case)) is False
        ):
            # Should have been caught by the validate_call?
            raise ValueError(
                "The supplied `new_run_from` is not valid. Please check the function description for more details."
            )

        entity_info_param = cls._get_user_requested_entity_info(
            current_project_id=project_info.asset_id, new_run_from=new_run_from
        )

        if root_type == RootType.GEOMETRY:
            root_asset = Geometry.from_cloud(meta.root_item_id, entity_info_param=entity_info_param)
        elif root_type == RootType.SURFACE_MESH:
            root_asset = SurfaceMeshV2.from_cloud(
                meta.root_item_id, entity_info_param=entity_info_param
            )
        elif root_type == RootType.VOLUME_MESH:
            root_asset = VolumeMeshV2.from_cloud(
                meta.root_item_id, entity_info_param=entity_info_param
            )
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

    def rename(self, new_name: str):
        """
        Rename the current project.

        Parameters
        ----------
        new_name : str
            The new name for the project.
        """
        RestApi(ProjectInterface.endpoint).patch(
            RenameAssetRequestV2(name=new_name).dict(), method=self.id
        )

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
        interpolate_to_mesh: VolumeMeshV2,
        run_async: bool,
        solver_version: str,
        use_beta_mesher: bool,
        use_geometry_AI: bool,
        raise_on_error: bool,
        tags: List[str],
        draft_only: bool,
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
        fork_from : Case, optional
            The parent case to fork from if fork (default is None).
        interpolate_to_mesh : VolumeMeshV2, optional
            If specified, forked case will interpolate parent case's results to this mesh before running solver.
        run_async : bool, optional
            Specifies whether the simulation should run asynchronously (default is True).
        use_beta_mesher : bool, optional
            Whether to use the beta mesher (default is None). Must be True when using Geometry AI.
        use_geometry_AI : bool, optional
            Whether to use the Geometry AI (default is False).
        raise_on_error: bool, optional
            Option to raise if submission error occurs
        tags: List[str], optional
            A list of tags to add to the target asset.
        draft_only: bool, optional
            Whether to only create and submit a draft and not run the simulation.

        Returns
        -------
        AssetOrResource
            The destination asset or the draft if `draft_only` is True.

        Raises
        ------
        Flow360ValueError
            If the simulation parameters lack required length unit information, or if the
            root asset (Geometry or VolumeMesh) is not initialized.
        """

        # pylint: disable=too-many-branches
        if use_beta_mesher is None:
            if use_geometry_AI is True:
                log.info("Beta mesher is enabled to use Geometry AI.")
                use_beta_mesher = True
            else:
                use_beta_mesher = False

        if use_geometry_AI is True and use_beta_mesher is False:
            raise Flow360ValueError("Enabling Geometry AI requires also enabling beta mesher.")

        params = set_up_params_for_uploading(
            params=params,
            root_asset=self._root_asset,
            length_unit=self.length_unit,
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=use_geometry_AI,
        )

        params, errors = validate_params_with_context(
            params=params,
            root_item_type=self.metadata.root_item_type.value,
            up_to=target._cloud_resource_type_name,
        )

        if errors is not None:
            log.error(
                f"Validation error found during local validation: {formatting_validation_errors(errors=errors)}"
            )
            if raise_on_error:
                raise ValueError("Submission terminated due to local validation error.")
            return None

        source_item_type = self.metadata.root_item_type.value if fork_from is None else "Case"
        start_from = kwargs.get("start_from", None)
        job_tags = kwargs.get("job_tags", None)

        all_tags = []

        if tags is not None:
            all_tags += tags
        if job_tags is not None:
            all_tags += job_tags

        draft = Draft.create(
            name=draft_name,
            project_id=self.metadata.id,
            source_item_id=self.metadata.root_item_id if fork_from is None else fork_from.id,
            source_item_type=source_item_type,
            solver_version=solver_version if solver_version else self.solver_version,
            fork_case=fork_from is not None,
            interpolation_volume_mesh_id=interpolate_to_mesh.id if interpolate_to_mesh else None,
            tags=all_tags,
        ).submit()

        params.pre_submit_summary()

        draft.update_simulation_params(params)

        if draft_only:
            # pylint: disable=import-outside-toplevel
            import click

            log.info("Draft submitted, copy the link to browser to view the draft:")
            # Not using log.info to avoid the link being wrapped and thus not clickable.
            click.secho(draft.web_url, fg="blue", underline=True)
            return draft

        try:
            destination_id = draft.run_up_to_target_asset(
                target,
                source_item_type=source_item_type,
                use_beta_mesher=use_beta_mesher,
                use_geometry_AI=use_geometry_AI,
                start_from=start_from,
            )
        except RuntimeError as exception:
            if raise_on_error:
                raise ValueError("Submission terminated due to validation error.") from exception
            return None

        self._project_webapi.patch(
            # pylint: disable=protected-access
            json={
                "lastOpenItemId": destination_id,
                "lastOpenItemType": target._cloud_resource_type_name,
            }
        )

        destination_obj = target.from_cloud(destination_id)

        log.info(f"Successfully submitted: {destination_obj.short_description()}")

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
        use_beta_mesher: bool = None,
        use_geometry_AI: bool = False,  # pylint: disable=invalid-name
        raise_on_error: bool = True,
        tags: List[str] = None,
        draft_only: bool = False,
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
        use_beta_mesher : bool, optional
            Whether to use the beta mesher (default is None). Must be True when using Geometry AI.
        use_geometry_AI : bool, optional
            Whether to use the Geometry AI (default is False).
        raise_on_error: bool, optional
            Option to raise if submission error occurs (default is True)
        tags: List[str], optional
            A list of tags to add to the generated surface mesh.
        draft_only: bool, optional
            Whether to only create and submit a draft and not generate the surface mesh.

        Raises
        ------
        Flow360ValueError
            If the root item type is not Geometry.

        Returns
        -------
        SurfaceMeshV2 | Draft
            The surface mesh asset or the draft if `draft_only` is True.
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
            interpolate_to_mesh=None,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=use_geometry_AI,
            raise_on_error=raise_on_error,
            tags=tags,
            draft_only=draft_only,
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
        use_beta_mesher: bool = None,
        use_geometry_AI: bool = False,  # pylint: disable=invalid-name
        raise_on_error: bool = True,
        tags: List[str] = None,
        draft_only: bool = False,
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
        use_beta_mesher : bool, optional
            Whether to use the beta mesher (default is None). Must be True when using Geometry AI.
        use_geometry_AI : bool, optional
            Whether to use the Geometry AI (default is False).
        raise_on_error: bool, optional
            Option to raise if submission error occurs (default is True)
        tags: List[str], optional
            A list of tags to add to the generated volume mesh.
        draft_only: bool, optional
            Whether to only create and submit a draft and not generate the volume mesh.

        Raises
        ------
        Flow360ValueError
            If the root item type is not Geometry.

        Returns
        -------
        VolumeMeshV2 | Draft
            The volume mesh asset or the draft if `draft_only` is True.
        """
        self._check_initialized()
        if (
            self.metadata.root_item_type is not RootType.GEOMETRY
            and self.metadata.root_item_type is not RootType.SURFACE_MESH
        ):
            raise Flow360ValueError(
                "Volume mesher can only be run by projects with a geometry or surface mesh root asset"
            )
        volume_mesh_or_draft = self._run(
            params=params,
            target=VolumeMeshV2,
            draft_name=name,
            run_async=run_async,
            fork_from=None,
            interpolate_to_mesh=None,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=use_geometry_AI,
            raise_on_error=raise_on_error,
            tags=tags,
            draft_only=draft_only,
            **kwargs,
        )
        if draft_only:
            draft = volume_mesh_or_draft
            return draft
        volume_mesh = volume_mesh_or_draft
        return volume_mesh

    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def run_case(
        self,
        params: SimulationParams,
        name: str = "Case",
        run_async: bool = True,
        fork_from: Optional[Case] = None,
        interpolate_to_mesh: Optional[VolumeMeshV2] = None,
        solver_version: str = None,
        use_beta_mesher: bool = None,
        use_geometry_AI: bool = False,  # pylint: disable=invalid-name
        raise_on_error: bool = True,
        tags: List[str] = None,
        draft_only: bool = False,
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
        interpolate_to_mesh : VolumeMeshV2, optional
            If specified, forked case will interpolate parent case results to this mesh before running solver.
        solver_version : str, optional
            Optional solver version to use during this run (defaults to the project solver version)
        use_beta_mesher : bool, optional
            Whether to use the beta mesher (default is None). Must be True when using Geometry AI.
        use_geometry_AI : bool, optional
            Whether to use the Geometry AI (default is False).
        raise_on_error: bool, optional
            Option to raise if submission error occurs (default is True)
        tags: List[str], optional
            A list of tags to add to the case.
        draft_only: bool, optional
            Whether to only create and submit a draft and not run the case.

        Returns
        -------
        Case | Draft
            The case asset or the draft if `draft_only` is True.
        """
        self._check_initialized()
        case_or_draft = self._run(
            params=params,
            target=Case,
            draft_name=name,
            run_async=run_async,
            fork_from=fork_from,
            interpolate_to_mesh=interpolate_to_mesh,
            solver_version=solver_version,
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=use_geometry_AI,
            raise_on_error=raise_on_error,
            tags=tags,
            draft_only=draft_only,
            **kwargs,
        )

        if draft_only:
            draft = case_or_draft
            return draft
        case = case_or_draft
        report_template = get_default_report_summary_template()
        report_template.create_in_cloud(
            name=f"{name}-summary",
            cases=[case],
            solver_version=solver_version if solver_version else self.solver_version,
        )
        return case
