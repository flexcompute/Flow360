"""
Support class and functions for project interface.
"""

import datetime
import os
from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.component.simulation import services
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    Slice,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.utils import (
    SUPPORTED_GEOMETRY_FILE_PATTERNS,
    MeshFileFormat,
    MeshNameParser,
    match_file_pattern,
    parse_datetime,
)
from flow360.exceptions import Flow360ConfigurationError, Flow360ValueError
from flow360.log import log


class InputFileModel(Flow360BaseModel):
    """Base model for input files creating projects"""

    file_names: Union[List[str], str] = pd.Field()

    def _check_files_existence(self) -> None:
        """
        Check if the file exists or not.
        """
        if isinstance(self.file_names, List):
            # pylint: disable = not-an-iterable
            for file_name in self.file_names:
                if not os.path.isfile(file_name):
                    raise ValueError(f"File {file_name} does not exist.")
        else:
            if not os.path.isfile(self.file_names):
                raise ValueError(f"File {self.file_names} does not exist.")


class GeometryFiles(InputFileModel):
    """Validation model to check if the given files are geometry files"""

    type_name: Literal["GeometryFile"] = pd.Field("GeometryFile", frozen=True)
    file_names: Union[List[str], str] = pd.Field()

    @pd.field_validator("file_names", mode="after")
    @classmethod
    def _validate_files(cls, value):
        if isinstance(value, str):
            if not match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, value):
                raise ValueError(
                    f"The given file: {value} is not a supported geometry file. "
                    f"Allowed file suffixes are: {SUPPORTED_GEOMETRY_FILE_PATTERNS}"
                )
        else:  # list
            for file in value:
                if not match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, file):
                    raise ValueError(
                        f"The given file: {file} is not a supported geometry file. "
                        f"Allowed file suffixes are: {SUPPORTED_GEOMETRY_FILE_PATTERNS}"
                    )
        return value


class SurfaceMeshFile(InputFileModel):
    """Validation model to check if the given file is a surface mesh file"""

    type_name: Literal["SurfaceMeshFile"] = pd.Field("SurfaceMeshFile", frozen=True)
    file_names: str = pd.Field()

    @pd.field_validator("file_names", mode="after")
    @classmethod
    def _validate_files(cls, value):
        try:
            parser = MeshNameParser(input_mesh_file=value)
        except Exception as e:
            raise ValueError(str(e)) from e
        if parser.is_valid_surface_mesh() or parser.is_valid_volume_mesh():
            # We support extracting surface mesh from volume mesh as well
            return value
        raise ValueError(
            f"The given mesh file {value} is not a valid surface mesh file. "
            f"Unsupported surface mesh file extensions: {parser.format.ext()}. "
            f"Supported: [{MeshFileFormat.UGRID.ext()},{MeshFileFormat.CGNS.ext()}, {MeshFileFormat.STL.ext()}]."
        )

    def _check_files_existence(self) -> None:
        """
        Check if the file exists or not. If it is ugrid file then check existence of mapbc file.
        """
        super()._check_files_existence()
        parser = MeshNameParser(input_mesh_file=self.file_names)
        if parser.is_ugrid():
            mapbc_file_name = parser.get_associated_mapbc_filename()
            if not os.path.isfile(mapbc_file_name):
                log.warning(
                    f"The mapbc file ({mapbc_file_name}) for {self.file_names} is not found"
                )


class VolumeMeshFile(InputFileModel):
    """Validation model to check if the given file is a volume mesh file"""

    type_name: Literal["VolumeMeshFile"] = pd.Field("VolumeMeshFile", frozen=True)
    file_names: str = pd.Field()

    @pd.field_validator("file_names", mode="after")
    @classmethod
    def _validate_files(cls, value):
        try:
            parser = MeshNameParser(input_mesh_file=value)
        except Exception as e:
            raise ValueError(str(e)) from e
        if parser.is_valid_volume_mesh():
            return value
        raise ValueError(
            f"The given mesh file {value} is not a valid volume mesh file. ",
            f"Unsupported volume mesh file extensions: {parser.format.ext()}. "
            f"Supported: [{MeshFileFormat.UGRID.ext()},{MeshFileFormat.CGNS.ext()}].",
        )


class AssetStatistics(pd.BaseModel):
    """Statistics for an asset"""

    count: int
    successCount: int
    runningCount: int
    divergedCount: int
    errorCount: int


class ProjectStatistics(pd.BaseModel):
    """Statistics for a project"""

    geometry: Optional[AssetStatistics] = pd.Field(None, alias="Geometry")
    surface_mesh: Optional[AssetStatistics] = pd.Field(None, alias="SurfaceMesh")
    volume_mesh: Optional[AssetStatistics] = pd.Field(None, alias="VolumeMesh")
    case: Optional[AssetStatistics] = pd.Field(None, alias="Case")


class ProjectInfo(pd.BaseModel):
    """Information about a project, retrieved from the projects get API"""

    name: str
    project_id: str = pd.Field(alias="id")
    tags: list[str] = pd.Field()
    description: str = pd.Field()
    statistics: ProjectStatistics = pd.Field()
    solver_version: Optional[str] = pd.Field(
        None, alias="solverVersion", description="If None then the project is from old database"
    )
    created_at: str = pd.Field(alias="createdAt")
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"] = pd.Field(
        alias="rootItemType"
    )

    @pd.computed_field
    @property
    def local_time_zone_created_time(self) -> datetime.datetime:
        """Convert string time to datetime obj and also convert to local time zone"""
        return parse_datetime(self.created_at)


class ProjectRecords(pd.BaseModel):
    """Holds all records of a user's project"""

    records: List[ProjectInfo] = pd.Field()

    def __str__(self):
        """Print out all info about the project"""
        if not self.records:
            output_str = "No matching projects found. Try skip naming patterns to show all."
            return output_str
        output_str = ">>> Projects sorted by creation time:\n"
        # pylint: disable=not-an-iterable
        for item in self.records:
            output_str += f" Name:         {item.name}\n"
            output_str += f" Created at:   {item.local_time_zone_created_time.strftime('%Y-%m-%d %H:%M %Z')}\n"
            output_str += f" Created with: {item.root_item_type}\n"
            output_str += f" ID:           {item.project_id}\n"
            output_str += (
                f" Link:         https://flow360.simulation.cloud/workbench/{item.project_id}\n"
            )
            if item.tags:
                output_str += f" Tags:         {item.tags}\n"
            if item.description:
                output_str += f" Description:  {item.description}\n"
            if item.statistics.geometry:
                output_str += f" Geometry count:     {item.statistics.geometry.count}\n"
            if item.statistics.surface_mesh:
                output_str += f" Surface Mesh count: {item.statistics.surface_mesh.count}\n"
            if item.statistics.volume_mesh:
                output_str += f" Volume Mesh count:  {item.statistics.volume_mesh.count}\n"
            if item.statistics.case:
                output_str += f" Case count:         {item.statistics.case.count}\n"

            output_str += "\n"
        return output_str


def show_projects_with_keyword_filter(search_keyword: str):
    """Show all projects with a keyword filter"""
    # pylint: disable=invalid-name
    MAX_DISPLAYABLE_ITEM_COUNT = 200
    MAX_SEARCHABLE_ITEM_COUNT = 1000
    _api = RestApi(ProjectInterface.endpoint, id=None)
    resp = _api.get(
        params={
            "page": "0",
            "size": MAX_SEARCHABLE_ITEM_COUNT,
            "filterKeywords": search_keyword,
            "sortFields": ["createdAt"],
            "sortDirections": ["asc"],
        }
    )

    all_projects = ProjectRecords.model_validate({"records": resp["records"]})
    log.info("%s", str(all_projects))

    if resp["total"] > MAX_DISPLAYABLE_ITEM_COUNT:
        log.warning(
            f"Total number of projects matching the keyword on the cloud is {resp['total']}, "
            f"but only the latest {MAX_DISPLAYABLE_ITEM_COUNT} will be displayed. "
        )
    log.info("Total number of matching projects on the cloud: %d", resp["total"])


def _replace_ghost_surfaces(params: SimulationParams):
    """
    When the `SimulationParam` is constructed with python script on the Python side, the ghost boundaries
    will be obtained by for example `automated_farfield.farfield` which returns :class:`GhostSurface`
    not :class:`GhostSphere`. This will not be recognized by the webUI causing the assigned farfield being
    removed by front end.
    """

    def _replace_the_ghost_surface(*, ghost_surface, ghost_entities_from_metadata):
        for item in ghost_entities_from_metadata:
            if item.name == ghost_surface.name:
                return item
        raise Flow360ConfigurationError(
            f"Ghost surface `{ghost_surface.name}` is used but likely won't be generated."
            " Please double check the use of ghost surfaces."
        )

    def _find_ghost_surfaces(*, model, ghost_entities_from_metadata):
        for field in model.__dict__.values():
            if isinstance(field, GhostSurface):
                # pylint: disable=protected-access
                field = _replace_the_ghost_surface(
                    ghost_surface=field, ghost_entities_from_metadata=ghost_entities_from_metadata
                )

            if isinstance(field, EntityList):
                if field.stored_entities:
                    for entity_index, _ in enumerate(field.stored_entities):
                        if isinstance(field.stored_entities[entity_index], GhostSurface):
                            field.stored_entities[entity_index] = _replace_the_ghost_surface(
                                ghost_surface=field.stored_entities[entity_index],
                                ghost_entities_from_metadata=ghost_entities_from_metadata,
                            )

            elif isinstance(field, list):
                for item in field:
                    if isinstance(item, Flow360BaseModel):
                        _find_ghost_surfaces(
                            model=item, ghost_entities_from_metadata=ghost_entities_from_metadata
                        )

            elif isinstance(field, Flow360BaseModel):
                _find_ghost_surfaces(
                    model=field, ghost_entities_from_metadata=ghost_entities_from_metadata
                )

    ghost_entities_from_metadata = (
        params.private_attribute_asset_cache.project_entity_info.ghost_entities
    )
    _find_ghost_surfaces(model=params, ghost_entities_from_metadata=ghost_entities_from_metadata)

    return params


def _set_up_params_persistent_entity_info(entity_info, params: SimulationParams):
    """
    Setting up the persistent entity info in params.
    1. Add the face/edge tags either by looking at the params' value or deduct the tags according to what is used.
    2. Reflect the changes to the existing persistent entities (like assigning tags or axis/centers).
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

    if isinstance(entity_info, GeometryEntityInfo):
        with model_attribute_unlock(entity_info, "face_group_tag"):
            entity_info.face_group_tag = _get_tag(entity_registry, Surface)
        with model_attribute_unlock(entity_info, "edge_group_tag"):
            entity_info.edge_group_tag = _get_tag(entity_registry, Edge)

    entity_info.update_persistent_entities(param_entity_registry=entity_registry)
    return entity_info


def _set_up_params_non_persistent_entity_info(entity_info, params: SimulationParams):
    """
    Setting up non-persistent entities (AKA draft entities) in params.
    Add the ones used to the entity info.
    """

    entity_registry = params.used_entity_registry
    # Creating draft entities
    for draft_type in [Box, Cylinder, Point, PointArray, Slice]:
        draft_entities = entity_registry.find_by_type(draft_type)
        for draft_entity in draft_entities:
            if draft_entity not in entity_info.draft_entities:
                entity_info.draft_entities.append(draft_entity)
    return entity_info


def _set_up_default_reference_geometry(params: SimulationParams, length_unit: LengthType):
    """
    Setting up the default reference geometry if not provided in params.
    Ensure the simulation.json contains the default settings other than None.
    """
    # pylint: disable=protected-access
    default_reference_geometry = services._get_default_reference_geometry(length_unit)
    if params.reference_geometry is None:
        params.reference_geometry = default_reference_geometry
        return params

    for field in params.reference_geometry.model_fields:
        if getattr(params.reference_geometry, field) is None:
            setattr(params.reference_geometry, field, getattr(default_reference_geometry, field))

    return params


def set_up_params_for_uploading(
    root_asset,
    length_unit: LengthType,
    params: SimulationParams,
    use_beta_mesher: bool,
):
    """
    Set up params before submitting the draft.
    """

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
        params.private_attribute_asset_cache.project_length_unit = length_unit

    with model_attribute_unlock(params.private_attribute_asset_cache, "use_inhouse_mesher"):
        params.private_attribute_asset_cache.use_inhouse_mesher = (
            use_beta_mesher if use_beta_mesher else False
        )

    entity_info = _set_up_params_persistent_entity_info(root_asset.entity_info, params)
    # Check if there are any new draft entities that have been added in the params by the user
    entity_info = _set_up_params_non_persistent_entity_info(entity_info, params)

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
        params.private_attribute_asset_cache.project_entity_info = entity_info
    # Replace the ghost surfaces in the SimulationParams by the real ghost ones from asset metadata.
    # This has to be done after `project_entity_info` is properly set.
    entity_info = _replace_ghost_surfaces(params)

    params = _set_up_default_reference_geometry(params, length_unit)

    return params


def validate_params_with_context(params, root_item_type, up_to):
    """Validate the simulation params with the simulation path."""

    # pylint: disable=protected-access
    validation_level = services._determine_validation_level(
        root_item_type=root_item_type, up_to=up_to
    )

    params, errors, _ = services.validate_model(
        params_as_dict=params.model_dump(),
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type=root_item_type,
        validation_level=validation_level,
    )

    return params, errors


def formatting_validation_errors(errors):
    """
    Format the validation errors to a human readable string.

    Example:
    --------
    Input: [{'type': 'missing', 'loc': ('meshing', 'defaults', 'boundary_layer_first_layer_thickness'),
            'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['VolumeMesh']},
            'url': 'https://errors.pydantic.dev/2.7/v/missing'}]

    Output: (1) Message: Field required | Location: meshing -> defaults -> boundary_layer_first_layer
    _thickness | Relevant for: ['VolumeMesh']
    """
    error_msg = ""
    for idx, error in enumerate(errors):
        error_msg += f"\n\t({idx+1}) Message: {error['msg']}"
        if error.get("loc") != ():
            location = " -> ".join([str(loc) for loc in error["loc"]])
            error_msg += f" | Location: {location}"
        if error.get("ctx") and error["ctx"].get("relevant_for"):
            error_msg += f" | Relevant for: {error['ctx']['relevant_for']}"
    return error_msg
