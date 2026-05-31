"""Utilities for entity registration and metadata updates in simulation models."""

from __future__ import annotations

from typing import Any

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.framework.unique_list import UniqueStringList
from flow360_schema.models.asset_cache import AssetCache
from flow360_schema.models.entities.base import _SurfaceEntityBase, _VolumeEntityBase


def find_instances(obj: Any, target_type: type[Any] | tuple[type[Any], ...]) -> list[Any]:
    """Recursively find items of the requested type within a Python object."""
    stack = [obj]
    seen_ids = set()
    results = []

    while stack:
        current = stack.pop()

        obj_id = id(current)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if isinstance(current, target_type):
            results.append(current)

        if isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
            continue

        if isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
            continue

        if hasattr(current, "__dict__"):
            stack.extend(vars(current).values())
            continue

        if hasattr(current, "__iter__") and not isinstance(current, (str, bytes)):
            stack.extend(iter(current))

    return list(results)


def register_entity_list(model: Flow360BaseModel, registry: EntityRegistry) -> None:
    """Register all entities reachable from a model into a registry."""
    known_frozen_hashes: set[str] = set()
    for field in model.__dict__.values():
        if isinstance(field, EntityBase):
            known_frozen_hashes = registry.fast_register(field, known_frozen_hashes)

        if isinstance(field, EntityList):
            for entity in field.stored_entities or []:
                known_frozen_hashes = registry.fast_register(entity, known_frozen_hashes)
            continue

        if isinstance(field, (list, tuple)):
            for item in field:
                if isinstance(item, EntityBase):
                    known_frozen_hashes = registry.fast_register(item, known_frozen_hashes)
                if isinstance(item, Flow360BaseModel):
                    register_entity_list(item, registry)
            continue

        if isinstance(field, Flow360BaseModel):
            register_entity_list(field, registry)


def _update_entity_full_name(
    model: Flow360BaseModel,
    target_entity_type: type[_SurfaceEntityBase] | type[_VolumeEntityBase],
    volume_mesh_meta_data: dict[str, Any],
) -> None:
    """Update surface or volume entity metadata from volume mesh metadata."""
    for field_name, field in model.__dict__.items():
        if isinstance(field, AssetCache):
            continue

        if isinstance(field, target_entity_type):
            field._update_entity_info_with_metadata(volume_mesh_meta_data)

        if isinstance(field, EntityList):
            added_entities = []
            for entity in field.stored_entities:
                if isinstance(entity, target_entity_type):
                    partial_additions = entity._update_entity_info_with_metadata(volume_mesh_meta_data)
                    if partial_additions is not None:
                        added_entities.extend(partial_additions)
            field.stored_entities.extend(added_entities)
            continue

        if isinstance(field, (list, tuple)):
            added_entities = []
            for item in field:
                if isinstance(item, target_entity_type):
                    partial_additions = item._update_entity_info_with_metadata(volume_mesh_meta_data)
                    if partial_additions is not None:
                        added_entities.extend(partial_additions)
                    continue
                if isinstance(item, Flow360BaseModel):
                    _update_entity_full_name(item, target_entity_type, volume_mesh_meta_data)

            if isinstance(field, list):
                field.extend(added_entities)
                continue
            if isinstance(field, tuple) and added_entities:
                model._force_set_attr(field_name, field + tuple(added_entities))
            continue

        if isinstance(field, Flow360BaseModel):
            _update_entity_full_name(field, target_entity_type, volume_mesh_meta_data)


def _update_zone_boundaries_with_metadata(
    registry: EntityRegistry,
    volume_mesh_meta_data: dict[str, Any],
) -> None:
    """Update zone boundary names on registered volume entities."""
    for volume_entity in [entity for view in registry.view_subclasses(_VolumeEntityBase) for entity in view._entities]:
        if volume_entity.name not in volume_mesh_meta_data["zones"]:
            continue
        volume_entity._force_set_attr(
            "private_attribute_zone_boundary_names",
            UniqueStringList(items=volume_mesh_meta_data["zones"][volume_entity.name]["boundaryNames"]),
        )


def _set_boundary_full_name_with_zone_name(
    registry: EntityRegistry,
    naming_pattern: str,
    give_zone_name: str,
) -> None:
    """Set boundary full names for matched surfaces that do not have one yet."""
    surfaces = [
        entity for entity in registry.find_by_naming_pattern(naming_pattern) if isinstance(entity, _SurfaceEntityBase)
    ]
    if not surfaces:
        return

    for surface in surfaces:
        if surface.private_attribute_full_name is not None:
            continue
        surface._force_set_attr("private_attribute_full_name", f"{give_zone_name}/{surface.name}")


def serialize_model_obj_to_id(model_obj: Flow360BaseModel) -> str:
    """Serialize a model object to its private attribute id."""
    private_attribute_id = getattr(model_obj, "private_attribute_id", None)
    if isinstance(private_attribute_id, str):
        return private_attribute_id
    raise ValueError(f"The model object {model_obj} cannot be serialized to id.")


__all__ = [
    "AssetCache",
    "find_instances",
    "register_entity_list",
    "_set_boundary_full_name_with_zone_name",
    "_update_entity_full_name",
    "_update_zone_boundaries_with_metadata",
    "serialize_model_obj_to_id",
]
