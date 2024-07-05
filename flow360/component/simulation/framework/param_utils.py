"""pre processing and post processing utilities for simulation parameters."""

from typing import Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import _model_attribute_unlock


class AssetCache(Flow360BaseModel):
    """
    Note:
    1. asset_entity_registry will be replacing/update the metadata-constructed registry of the asset when loading it.
    """

    asset_entity_registry: EntityRegistry = pd.Field(EntityRegistry(), frozen=True)
    project_length_unit: Optional[LengthType.Positive] = pd.Field(None, frozen=True)


def recursive_register_entity_list(model: Flow360BaseModel, registry: EntityRegistry) -> None:
    """
    Recursively registers entities within a Flow360BaseModel instance to an EntityRegistry.

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
    for field in model.__dict__.values():
        if isinstance(field, EntityBase):
            registry.register(field)

        if isinstance(field, EntityList):
            # pylint: disable=protected-access
            expanded_entities = field._get_expanded_entities(
                supplied_registry=None, expect_supplied_registry=False, create_hard_copy=False
            )
            for entity in expanded_entities if expanded_entities else []:
                registry.register(entity)

        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Flow360BaseModel):
                    recursive_register_entity_list(item, registry)

        elif isinstance(field, Flow360BaseModel):
            recursive_register_entity_list(field, registry)


def _recursive_update_zone_name_in_surface_with_metadata(
    model: Flow360BaseModel, volume_mesh_meta_data: dict
):
    """
    Update the zone info from volume mesh
    """
    for field in model.__dict__.values():
        if isinstance(field, Surface):
            # pylint: disable=protected-access
            field._set_boundary_full_name_from_metadata(volume_mesh_meta_data)

        if isinstance(field, EntityList):
            # pylint: disable=protected-access
            expanded_entities = field._get_expanded_entities(
                supplied_registry=None, expect_supplied_registry=False, create_hard_copy=False
            )
            for entity in expanded_entities if expanded_entities else []:
                if isinstance(entity, Surface):
                    entity._set_boundary_full_name_from_metadata(volume_mesh_meta_data)

        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Flow360BaseModel):
                    _recursive_update_zone_name_in_surface_with_metadata(
                        item, volume_mesh_meta_data
                    )

        elif isinstance(field, Flow360BaseModel):
            _recursive_update_zone_name_in_surface_with_metadata(field, volume_mesh_meta_data)


def _set_boundary_full_name_with_zone_name(
    registry: EntityRegistry, naming_pattern: str, give_zone_name: str
) -> None:
    """Get the entity registry."""
    if registry.find_by_naming_pattern(naming_pattern):
        for surface in registry.find_by_naming_pattern(naming_pattern):
            with _model_attribute_unlock(surface, "private_attribute_full_name"):
                surface.private_attribute_full_name = f"{give_zone_name}/{surface.name}"
