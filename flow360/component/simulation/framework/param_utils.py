"""pre processing and post processing utilities for simulation parameters."""

import re
from typing import TYPE_CHECKING, Annotated, List, Optional, Union, get_args

import pydantic as pd

from flow360.component.simulation.entity_info import (
    GeometryEntityInfo,
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.unique_list import UniqueStringList
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
)
from flow360.component.simulation.meshing_param.volume_params import (
    RotationCylinder,
    RotationVolume,
)
from flow360.component.simulation.models.surface_models import (
    SurfaceModelTypes,
    Wall,
)
from flow360.component.simulation.primitives import (
    Surface,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import (
    VariableContextInfo,
    update_global_context,
)
from flow360.component.simulation.utils import model_attribute_unlock

if TYPE_CHECKING:
    from flow360.component.simulation.simulation_params import SimulationParams

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
        None,
        description="List of user variables that are used in all the `Expression` instances.",
    )
    used_selectors: Optional[List[dict]] = pd.Field(
        None,
        description="Collected entity selectors for token reference.",
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
        flow360_unit_system=None,
    ) -> Flow360BaseModel:
        exclude_asset_cache = exclude + ["variable_context", "selectors"]
        return super().preprocess(
            params=params,
            exclude=exclude_asset_cache,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
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


def _update_rotating_entity_names_with_metadata(
    params: "SimulationParams", volume_mesh_meta_data: dict
) -> tuple[dict, set, dict]:
    """
    Update entity names to point to __rotating patches.

    For each RotationVolume:
    1. Check if __rotating patches exist in metadata for enclosed_entities
    2. Update enclosed_entities to reference __rotating patches
    3. Update stationary_enclosed_entities to reference __rotating patches

    Returns
    -------
    tuple[dict, set, dict]
        A tuple containing:
        - Mapping of original_boundary_name -> rotating_boundary_name for all entities with __rotating patches.
        - Set of rotating_boundary_name values that are in stationary_enclosed_entities.
        - Mapping of rotating_boundary_name -> updated Surface entity to avoid creating duplicates.
    """
    # Get volume zones from params
    if params.meshing is None:
        return {}, set(), {}

    volume_zones = None
    if isinstance(params.meshing, MeshingParams):
        volume_zones = params.meshing.volume_zones
    elif (
        isinstance(params.meshing, ModularMeshingWorkflow)
        and params.meshing.volume_meshing is not None
    ):
        volume_zones = params.meshing.zones

    if volume_zones is None:
        return {}, set(), {}

    # Find all RotationVolume instances
    rotation_volumes = [zone for zone in volume_zones if isinstance(zone, RotationVolume)]
    if len(rotation_volumes) == 0:
        return {}, set(), {}

    # Track mapping: original_boundary_name -> rotating_boundary_name for all entities with __rotating patches
    # This is used to copy models later
    original_to_rotating_map = {}
    # Track which boundaries are in stationary_enclosed_entities (for velocity=0 setting)
    stationary_boundaries = set()
    # Track updated entities: rotating_boundary_name -> updated Surface entity
    # This avoids creating duplicate Surface entities in the model update function
    rotating_boundary_to_entity_map = {}

    # Process each RotationVolume
    for rotation_volume in rotation_volumes:
        # Get the entity name from the rotation volume's entity
        # The zone name may have prefixes/suffixes (e.g., "rotatingBlock-{entity_name}")
        entity_name = None
        zone_name = None
        if rotation_volume.entities and rotation_volume.entities.stored_entities:
            entity = rotation_volume.entities.stored_entities[0]
            entity_name = entity.name

            # Search through zones to find the one matching this entity
            # Zone name may have any prefix/suffix, so check if entity_name appears in zone_name
            for candidate_zone_name in volume_mesh_meta_data.get("zones", {}).keys():
                # Check if entity_name is contained in zone_name
                # This handles cases like "rotatingBlock-{entity_name}", "{entity_name}-suffix", etc.
                if entity_name in candidate_zone_name:
                    zone_name = candidate_zone_name
                    break

        if zone_name is None:
            continue

        # Get zone metadata
        zone_meta = volume_mesh_meta_data.get("zones", {}).get(zone_name)
        if zone_meta is None:
            continue

        boundary_names = zone_meta.get("boundaryNames", [])

        # Track which entities are in stationary_enclosed_entities
        stationary_entity_names = set()
        if rotation_volume.stationary_enclosed_entities is not None:
            for stationary_entity in rotation_volume.stationary_enclosed_entities.stored_entities:
                if isinstance(stationary_entity, Surface):
                    stationary_entity_names.add(stationary_entity.name)

        # Process enclosed_entities
        if rotation_volume.enclosed_entities is not None:
            for enclosed_entity in rotation_volume.enclosed_entities.stored_entities:
                if not isinstance(enclosed_entity, Surface):
                    continue

                original_boundary_name = enclosed_entity.full_name

                # Extract the base boundary name (without zone prefix)
                # Pattern: zone_name/boundary_name or just boundary_name
                base_boundary_name = original_boundary_name
                if "/" in original_boundary_name:
                    base_boundary_name = original_boundary_name.split("/", 1)[1]

                # Look for __rotating patch: zone_name/base_boundary_name__rotating_zone_name
                # Pattern: {zone_name}/{base_boundary_name}__rotating_{zone_name}
                rotating_pattern = (
                    re.escape(zone_name)
                    + r"/"
                    + re.escape(base_boundary_name)
                    + r"__rotating_"
                    + re.escape(zone_name)
                )
                rotating_boundary_name = None
                for boundary_name in boundary_names:
                    if re.fullmatch(rotating_pattern, boundary_name):
                        rotating_boundary_name = boundary_name
                        break

                if rotating_boundary_name is None:
                    continue

                # Extract base name from rotating boundary (without zone prefix)
                rotating_base_name = (
                    rotating_boundary_name.split("/")[-1]
                    if "/" in rotating_boundary_name
                    else rotating_boundary_name
                )

                # Store original name before updating (needed for stationary check)
                original_entity_name = enclosed_entity.name

                # Update the entity to point to the __rotating patch
                # Since name is frozen, we need to create a new entity with updated name
                updated_entity = enclosed_entity.copy(
                    update={
                        "name": rotating_base_name,
                        "private_attribute_full_name": rotating_boundary_name,
                    }
                )
                # Replace the entity in the list
                entity_index = rotation_volume.enclosed_entities.stored_entities.index(
                    enclosed_entity
                )
                rotation_volume.enclosed_entities.stored_entities[entity_index] = updated_entity

                # Track this mapping for model copying (for all boundaries with __rotating patches)
                original_to_rotating_map[original_boundary_name] = rotating_boundary_name
                # Track the updated entity to avoid creating duplicates in model update
                rotating_boundary_to_entity_map[rotating_boundary_name] = updated_entity

                # Check if this entity is in stationary_enclosed_entities
                if original_entity_name in stationary_entity_names:
                    # Mark this boundary as stationary (for velocity=0 setting)
                    stationary_boundaries.add(rotating_boundary_name)

                    # Update stationary entity to also point to __rotating patch
                    if rotation_volume.stationary_enclosed_entities:
                        for idx, stationary_entity in enumerate(
                            rotation_volume.stationary_enclosed_entities.stored_entities
                        ):
                            # these are all Surface by construction
                            if stationary_entity.name == original_entity_name:
                                updated_stationary_entity = stationary_entity.copy(
                                    update={
                                        "name": rotating_base_name,
                                        "private_attribute_full_name": rotating_boundary_name,
                                    }
                                )
                                rotation_volume.stationary_enclosed_entities.stored_entities[
                                    idx
                                ] = updated_stationary_entity
                                break

    return (
        original_to_rotating_map,
        stationary_boundaries,
        rotating_boundary_to_entity_map,
    )


def _update_rotating_models_with_metadata(
    params: "SimulationParams",
    original_to_rotating_map: dict,
    stationary_boundaries: set,
    rotating_boundary_to_entity_map: dict,
) -> None:
    """
    Copy boundary condition models from original patches to __rotating patches.

    For all boundaries with __rotating patches, copies models from the original
    boundary to the __rotating boundary. For boundaries in stationary_enclosed_entities,
    also sets velocity to zero for Wall models.

    Parameters
    ----------
    params : SimulationParams
        The simulation parameters to update.
    original_to_rotating_map : dict
        Mapping of original_boundary_name -> rotating_boundary_name for all entities with __rotating patches.
    stationary_boundaries : set
        Set of rotating_boundary_name values that are in stationary_enclosed_entities.
    """
    if params.models is None or len(original_to_rotating_map) == 0:
        return

    models_to_add = []
    for model in params.models:
        # only Wall boundaries need special treatment
        if not isinstance(model, Wall):
            continue
        if not model.entities:
            continue

        # Check if this model references any of the original boundaries
        model_entities = model.entities.stored_entities
        rotating_surfaces_stationary = []
        rotating_surfaces_non_stationary = []

        for entity in model_entities:
            if not isinstance(entity, Surface):
                continue

            entity_full_name = entity.full_name

            # Check if this entity's boundary needs model copying
            if entity_full_name in original_to_rotating_map:
                rotating_boundary_name = original_to_rotating_map[entity_full_name]
                # Reuse the updated entity from enclosed_entities if available, otherwise create new
                assert rotating_boundary_name in rotating_boundary_to_entity_map
                rotating_surface = rotating_boundary_to_entity_map[rotating_boundary_name]

                # Separate into stationary and non-stationary
                if rotating_boundary_name in stationary_boundaries:
                    rotating_surfaces_stationary.append(rotating_surface)
                else:
                    rotating_surfaces_non_stationary.append(rotating_surface)

        # Create separate models for stationary and non-stationary entities
        if rotating_surfaces_stationary:
            update_dict = {"velocity": ("0", "0", "0")}
            new_model = model.copy(update=update_dict)
            new_model.entities = rotating_surfaces_stationary
            models_to_add.append(new_model)

        # Non-stationary entities keep original velocity settings
        if rotating_surfaces_non_stationary:
            new_model = model.copy()
            new_model.entities = rotating_surfaces_non_stationary
            models_to_add.append(new_model)

    # Add the new models to params.models
    if models_to_add:
        params.models.extend(models_to_add)


def _update_rotating_boundaries_with_metadata(
    params: "SimulationParams", volume_mesh_meta_data: dict
):
    """
    Update rotating boundaries with metadata from volume mesh.

    This function orchestrates the update process by:
    1. Updating entity names to point to __rotating patches
    2. Copying boundary condition models for all boundaries with __rotating patches
    """
    # Update entity names and get mapping for model copying
    (
        original_to_rotating_map,
        stationary_boundaries,
        rotating_boundary_to_entity_map,
    ) = _update_rotating_entity_names_with_metadata(params, volume_mesh_meta_data)

    # Copy models for all boundaries with __rotating patches
    _update_rotating_models_with_metadata(
        params,
        original_to_rotating_map,
        stationary_boundaries,
        rotating_boundary_to_entity_map,
    )
