"""pre processing and post processing utilities for simulation parameters."""

# pylint: disable=no-member

from typing import Union

from flow360_schema.models.asset_cache import AssetCache

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.unique_list import UniqueStringList
from flow360.component.simulation.primitives import (
    _SurfaceEntityBase,
    _VolumeEntityBase,
)


def find_instances(obj, target_type):
    """Recursively find items of target_type within a python object"""
    stack = [obj]
    seen_ids = set()
    results = set()

    while stack:
        current = stack.pop()

        obj_id = id(current)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if isinstance(current, target_type):
            results.add(current)

        if isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())

        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)

        elif hasattr(current, "__dict__"):
            stack.extend(vars(current).values())

        elif hasattr(current, "__iter__") and not isinstance(current, (str, bytes)):
            try:
                stack.extend(iter(current))
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # skip problematic iterables

    return list(results)


def register_entity_list(model: Flow360BaseModel, registry: EntityRegistry) -> None:
    """
    Registers entities used/occurred in a Flow360BaseModel instance to an EntityRegistry.

    This function iterates through the attributes of the given model. If an attribute is an
    EntityList, it retrieves the expanded entities and registers each entity in the registry.
    If an attribute is a list and contains instances of Flow360BaseModel, it recursively
    registers the entities within those instances.

    Args:
        model (Flow360BaseModel): The model containing entities to be registered.
        registry (EntityRegistry): The registry where entities will be registered.

    Returns:
        None
    """
    known_frozen_hashes = set()
    for field in model.__dict__.values():
        if isinstance(field, EntityBase):
            known_frozen_hashes = registry.fast_register(field, known_frozen_hashes)

        if isinstance(field, EntityList):
            for entity in field.stored_entities if field.stored_entities else []:
                known_frozen_hashes = registry.fast_register(entity, known_frozen_hashes)

        elif isinstance(field, (list, tuple)):
            for item in field:
                if isinstance(item, Flow360BaseModel):
                    register_entity_list(item, registry)

        elif isinstance(field, Flow360BaseModel):
            register_entity_list(field, registry)


# pylint: disable=too-many-branches
def _update_entity_full_name(
    model: Flow360BaseModel,
    target_entity_type: Union[type[_SurfaceEntityBase], type[_VolumeEntityBase]],
    volume_mesh_meta_data: dict,
):
    """
    Update Surface/Boundary with zone name from volume mesh metadata.
    """
    for field in model.__dict__.values():
        # Skip the AssetCache since updating there makes no difference
        if isinstance(field, AssetCache):
            continue

        if isinstance(field, target_entity_type):
            # pylint: disable=protected-access
            field._update_entity_info_with_metadata(volume_mesh_meta_data)

        if isinstance(field, EntityList):
            added_entities = []
            for entity in field.stored_entities:
                if isinstance(entity, target_entity_type):
                    # pylint: disable=protected-access
                    partial_additions = entity._update_entity_info_with_metadata(
                        volume_mesh_meta_data
                    )
                    if partial_additions is not None:
                        added_entities.extend(partial_additions)
            field.stored_entities.extend(added_entities)

        elif isinstance(field, (list, tuple)):
            added_entities = []
            for item in field:
                if isinstance(item, target_entity_type):
                    partial_additions = (
                        item._update_entity_info_with_metadata(  # pylint: disable=protected-access
                            volume_mesh_meta_data
                        )
                    )
                    if partial_additions is not None:
                        added_entities.extend(partial_additions)
                elif isinstance(item, Flow360BaseModel):
                    _update_entity_full_name(item, target_entity_type, volume_mesh_meta_data)

            if isinstance(field, list):
                field.extend(added_entities)
            if isinstance(field, tuple):
                field += tuple(added_entities)

        elif isinstance(field, Flow360BaseModel):
            _update_entity_full_name(field, target_entity_type, volume_mesh_meta_data)


def _update_zone_boundaries_with_metadata(
    registry: EntityRegistry, volume_mesh_meta_data: dict
) -> None:
    """Update zone boundaries with volume mesh metadata."""
    for volume_entity in [
        # pylint: disable=protected-access
        entity
        for view in registry.view_subclasses(_VolumeEntityBase)
        for entity in view._entities
    ]:
        if volume_entity.name in volume_mesh_meta_data["zones"]:
            volume_entity._force_set_attr(  # pylint:disable=protected-access
                "private_attribute_zone_boundary_names",
                UniqueStringList(
                    items=volume_mesh_meta_data["zones"][volume_entity.name]["boundaryNames"]
                ),
            )


def _set_boundary_full_name_with_zone_name(
    registry: EntityRegistry, naming_pattern: str, give_zone_name: str
) -> None:
    """Set the full name of surfaces that does not have full name specified."""
    if registry.find_by_naming_pattern(naming_pattern):
        for surface in registry.find_by_naming_pattern(naming_pattern):
            if surface.private_attribute_full_name is not None:
                # This indicates that full name has been set by mesh metadata because that and this are the
                # only two places we set the full name.
                # mesh meta data takes precedence as it is the most reliable source.
                # Note: Currently automated farfield assumes zone name to be "fluid" but the other mesher has "1".
                # Note: We need to figure out how to handle this. Otherwise this may result in wrong
                # Note: zone name getting prepended.
                continue
            surface._force_set_attr(  # pylint:disable=protected-access
                "private_attribute_full_name", f"{give_zone_name}/{surface.name}"
            )


def serialize_model_obj_to_id(model_obj: Flow360BaseModel) -> str:
    """Serialize a model object to its id."""
    if hasattr(model_obj, "private_attribute_id"):
        return model_obj.private_attribute_id
    raise ValueError(f"The model object {model_obj} cannot be serialized to id.")
