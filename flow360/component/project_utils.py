"""
Support class and functions for project interface.
"""

from typing import Union

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
from flow360.exceptions import Flow360ConfigurationError, Flow360ValueError


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
            f"Unknown ghost surface with name `{ghost_surface.name}` found."
            "Please double check the use of ghost surfaces."
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


def set_up_params_for_uploading(
    root_asset,
    length_unit: LengthType,
    params: SimulationParams,
):
    """
    Set up params before submitting the draft.
    """

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
        params.private_attribute_asset_cache.project_length_unit = length_unit

    entity_info = _set_up_params_persistent_entity_info(root_asset.entity_info, params)
    # Check if there are any new draft entities that have been added in the params by the user
    entity_info = _set_up_params_non_persistent_entity_info(entity_info, params)

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
        params.private_attribute_asset_cache.project_entity_info = entity_info
    # Replace the ghost surfaces in the SimulationParams by the real ghost ones from asset metadata.
    # This has to be done after `project_entity_info` is properly set.
    entity_info = _replace_ghost_surfaces(params)

    return params


def validate_params_with_context(params, root_item_type, up_to):
    """Validate the simulation params with the simulation path."""

    # pylint: disable=protected-access
    validation_level = services._determine_validation_level(
        root_item_type=root_item_type, up_to=up_to
    )

    params, errors, _ = services.validate_model(
        params_as_dict=params.model_dump(),
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
