"""
Support class and functions for project interface.
"""

import datetime
from typing import List, Literal, Optional

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.component.simulation import services
from flow360.component.simulation.entity_info import EntityInfoModel
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.primitives import (
    Box,
    CustomVolume,
    Cylinder,
    Edge,
    GeometryBodyGroup,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import save_user_variables
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.utils import parse_datetime
from flow360.exceptions import Flow360ConfigurationError
from flow360.log import log


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


def get_project_records(
    search_keyword: str, tags: Optional[List[str]] = None
) -> tuple[ProjectRecords, int]:
    """Get all projects with a keyword filter"""
    # pylint: disable=invalid-name
    MAX_SEARCHABLE_ITEM_COUNT = 1000
    _api = RestApi(ProjectInterface.endpoint, id=None)
    resp = _api.get(
        params={
            "page": "0",
            "size": MAX_SEARCHABLE_ITEM_COUNT,
            "filterKeywords": search_keyword,
            "filterTags": tags,
            "sortFields": ["createdAt"],
            "sortDirections": ["asc"],
        }
    )

    all_projects = ProjectRecords.model_validate({"records": resp["records"]})
    num_of_projects = resp["total"]

    return all_projects, num_of_projects


def show_projects_with_keyword_filter(search_keyword: str):
    """Show all projects with a keyword filter"""
    # pylint: disable=invalid-name
    MAX_DISPLAYABLE_ITEM_COUNT = 200
    all_projects, num_of_projects = get_project_records(search_keyword)
    log.info("%s", str(all_projects))

    if num_of_projects > MAX_DISPLAYABLE_ITEM_COUNT:
        log.warning(
            f"Total number of projects matching the keyword on the cloud is {num_of_projects}, "
            f"but only the latest {MAX_DISPLAYABLE_ITEM_COUNT} will be displayed. "
        )
    log.info("Total number of matching projects on the cloud: %d", num_of_projects)


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

            elif isinstance(field, (list, tuple)):
                for item in field:
                    if isinstance(item, GhostSurface):
                        # pylint: disable=protected-access
                        item = _replace_the_ghost_surface(
                            ghost_surface=item,
                            ghost_entities_from_metadata=ghost_entities_from_metadata,
                        )
                    elif isinstance(item, Flow360BaseModel):
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


def _set_up_params_non_persistent_entity_info(entity_info, params: SimulationParams):
    """
    Setting up non-persistent entities (AKA draft entities) in params.
    Add the ones used to the entity info.
    """

    entity_registry = params.used_entity_registry
    # Creating draft entities
    for draft_type in [Box, Cylinder, Point, PointArray, PointArray2D, Slice, CustomVolume]:
        draft_entities = entity_registry.find_by_type(draft_type)
        for draft_entity in draft_entities:
            if draft_entity not in entity_info.draft_entities:
                entity_info.draft_entities.append(draft_entity)
    return entity_info


def _update_entity_grouping_tags(entity_info, params: SimulationParams) -> EntityInfoModel:
    """
    Update the entity grouping tags in params to resolve possible conflicts
    between the SimulationParams and the root asset.This
    """

    def _get_used_tags(model: Flow360BaseModel, target_entity_type, used_tags: set):
        for field in model.__dict__.values():
            # Skip the AssetCache since the asset cache is exactly what we want to update later.

            if isinstance(field, AssetCache):
                continue

            if isinstance(field, target_entity_type):
                used_tags.add(field.private_attribute_tag_key)

            if isinstance(field, EntityList):
                for entity in field.stored_entities:
                    if isinstance(entity, target_entity_type):
                        used_tags.add(entity.private_attribute_tag_key)

            elif isinstance(field, (list, tuple)):
                for item in field:
                    if isinstance(item, target_entity_type):
                        used_tags.add(item.private_attribute_tag_key)
                    elif isinstance(item, Flow360BaseModel):
                        _get_used_tags(item, target_entity_type, used_tags)

            elif isinstance(field, Flow360BaseModel):
                _get_used_tags(field, target_entity_type, used_tags)

    if entity_info.type_name != "GeometryEntityInfo":
        return entity_info
    # pylint: disable=protected-access
    entity_types = [
        (Surface, "face_group_tag", entity_info._get_default_grouping_tag("face")),
    ]

    if entity_info.edge_ids:
        entity_types.append((Edge, "edge_group_tag", entity_info._get_default_grouping_tag("edge")))

    if entity_info.body_ids:
        entity_types.append(
            (GeometryBodyGroup, "body_group_tag", entity_info._get_default_grouping_tag("body"))
        )

    for entity_type, entity_grouping_tags, default_grouping_tag in entity_types:
        used_tags = set()
        _get_used_tags(params, entity_type, used_tags)

        if None in used_tags:
            used_tags.remove(None)

        used_tags = sorted(list(used_tags))
        current_tag = getattr(entity_info, entity_grouping_tags)
        if len(used_tags) == 1 and current_tag != used_tags[0]:
            if current_tag == default_grouping_tag:
                log.warning(
                    f"Auto reset the grouping to the one in the SimulationParams ({used_tags[0]})."
                )
                with model_attribute_unlock(entity_info, entity_grouping_tags):
                    setattr(entity_info, entity_grouping_tags, used_tags[0])
            else:
                # User specified new grouping
                raise Flow360ConfigurationError(
                    f"Conflicting entity ({entity_type.__name__}) grouping tags found "
                    f"in the SimulationParams ({used_tags}) and "
                    f"the root asset ({current_tag})."
                )

        if len(used_tags) > 1:
            raise Flow360ConfigurationError(
                f"Multiple entity ({entity_type.__name__}) grouping tags found "
                f"in the SimulationParams ({used_tags})."
            )

    return entity_info


def _set_up_default_geometry_accuracy(
    root_asset,
    params: SimulationParams,
    use_geometry_AI: bool,  # pylint: disable=invalid-name
):
    """
    Set up the default geometry accuracy in params if not set by the user.
    """
    if not use_geometry_AI:
        return params
    if root_asset.default_settings.get("geometry_accuracy") is None:
        return params
    if not params.meshing.defaults.geometry_accuracy:
        params.meshing.defaults.geometry_accuracy = root_asset.default_settings["geometry_accuracy"]
    return params


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

    for field in params.reference_geometry.__class__.model_fields:
        if getattr(params.reference_geometry, field) is None:
            setattr(params.reference_geometry, field, getattr(default_reference_geometry, field))

    return params


def set_up_params_for_uploading(
    root_asset,
    length_unit: LengthType,
    params: SimulationParams,
    use_beta_mesher: bool,
    use_geometry_AI: bool,  # pylint: disable=invalid-name
) -> SimulationParams:
    """
    Set up params before submitting the draft.
    """

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
        params.private_attribute_asset_cache.project_length_unit = length_unit

    with model_attribute_unlock(params.private_attribute_asset_cache, "use_inhouse_mesher"):
        params.private_attribute_asset_cache.use_inhouse_mesher = (
            use_beta_mesher if use_beta_mesher else False
        )

    with model_attribute_unlock(params.private_attribute_asset_cache, "use_geometry_AI"):
        params.private_attribute_asset_cache.use_geometry_AI = (
            use_geometry_AI if use_geometry_AI else False
        )

    # User may have made modifications to the entities which is recorded in asset's entity registry
    # We need to reflect these changes.
    root_asset.entity_info.update_persistent_entities(
        asset_entity_registry=root_asset.internal_registry
    )

    # Check if there are any new draft entities that have been added in the params by the user
    entity_info = _set_up_params_non_persistent_entity_info(root_asset.entity_info, params)

    # If the customer just load the param without re-specify the same set of entity grouping tags,
    # we need to update the entity grouping tags to the ones in the SimulationParams.
    entity_info = _update_entity_grouping_tags(entity_info, params)

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
        params.private_attribute_asset_cache.project_entity_info = entity_info
    # Replace the ghost surfaces in the SimulationParams by the real ghost ones from asset metadata.
    # This has to be done after `project_entity_info` is properly set.
    params = _replace_ghost_surfaces(params)
    params = _set_up_default_geometry_accuracy(root_asset, params, use_geometry_AI)

    params = _set_up_default_reference_geometry(params, length_unit)

    # Convert all reference of UserVariables to VariableToken
    params = save_user_variables(params)

    return params


def validate_params_with_context(params, root_item_type, up_to):
    """Validate the simulation params with the simulation path."""

    # pylint: disable=protected-access
    validation_level = services._determine_validation_level(
        root_item_type=root_item_type, up_to=up_to
    )

    params, errors, _ = services.validate_model(
        params_as_dict=params.model_dump(mode="json", exclude_none=True),
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type=root_item_type,
        validation_level=validation_level,
    )

    return params, errors
