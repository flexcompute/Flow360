"""pre processing and post processing utilities for simulation parameters."""

from typing import Annotated, List, Optional, Union

import pydantic as pd

from flow360.component.simulation.entity_info import (
    GeometryEntityInfo,
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360.component.simulation.framework.base_model import (
    Flow360BaseModel,
    RegistryLookup,
)
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.unique_list import UniqueStringList
from flow360.component.simulation.primitives import (
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import (
    VariableContextInfo,
    update_global_context,
)
from flow360.component.simulation.utils import model_attribute_unlock

VariableContextList = Annotated[
    List[VariableContextInfo],
    pd.AfterValidator(update_global_context),
]


class AssetCache(Flow360BaseModel):
    """
    Cached info from the project asset.
    """

    # pylint: disable=no-member
    project_length_unit: Optional[LengthType.Positive] = pd.Field(None, frozen=True)
    project_entity_info: Optional[
        Union[GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo]
    ] = pd.Field(None, frozen=True, discriminator="type_name")
    use_inhouse_mesher: bool = pd.Field(
        False,
        description="Flag whether user requested the use of inhouse surface and volume mesher.",
    )
    use_geometry_AI: bool = pd.Field(
        False, description="Flag whether user requested the use of GAI."
    )
    variable_context: Optional[VariableContextList] = pd.Field(
        None, description="List of user variables that are used in all the `Expression` instances."
    )

    @property
    def boundaries(self):
        """
        Get all boundaries (not just names) from the cached entity info.
        """
        if self.project_entity_info is None:
            return None
        return self.project_entity_info.get_boundaries()

    def preprocess(
        self,
        *,
        params=None,
        exclude: List[str] = None,
        required_by: List[str] = None,
        registry_lookup: RegistryLookup = None,
    ) -> Flow360BaseModel:
        exclude_asset_cache = exclude + ["variable_context"]
        return super().preprocess(
            params=params,
            exclude=exclude_asset_cache,
            required_by=required_by,
            registry_lookup=registry_lookup,
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
            # pylint: disable=protected-access
            expanded_entities = field._get_expanded_entities(create_hard_copy=False)
            for entity in expanded_entities if expanded_entities else []:
                known_frozen_hashes = registry.fast_register(entity, known_frozen_hashes)

        elif isinstance(field, (list, tuple)):
            for item in field:
                if isinstance(item, Flow360BaseModel):
                    register_entity_list(item, registry)

        elif isinstance(field, Flow360BaseModel):
            register_entity_list(field, registry)


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
            for entity in field.stored_entities:
                if isinstance(entity, target_entity_type):
                    # pylint: disable=protected-access
                    entity._update_entity_info_with_metadata(volume_mesh_meta_data)

        elif isinstance(field, (list, tuple)):
            for item in field:
                if isinstance(item, target_entity_type):
                    item._update_entity_info_with_metadata(  # pylint: disable=protected-access
                        volume_mesh_meta_data
                    )
                elif isinstance(item, Flow360BaseModel):
                    _update_entity_full_name(item, target_entity_type, volume_mesh_meta_data)

        elif isinstance(field, Flow360BaseModel):
            _update_entity_full_name(field, target_entity_type, volume_mesh_meta_data)


def _update_zone_boundaries_with_metadata(
    registry: EntityRegistry, volume_mesh_meta_data: dict
) -> None:
    """Update zone boundaries with volume mesh metadata."""
    for volume_entity in registry.get_bucket(by_type=_VolumeEntityBase).entities:
        if volume_entity.name in volume_mesh_meta_data["zones"]:
            with model_attribute_unlock(volume_entity, "private_attribute_zone_boundary_names"):
                volume_entity.private_attribute_zone_boundary_names = UniqueStringList(
                    items=volume_mesh_meta_data["zones"][volume_entity.name]["boundaryNames"]
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
            with model_attribute_unlock(surface, "private_attribute_full_name"):
                surface.private_attribute_full_name = f"{give_zone_name}/{surface.name}"
