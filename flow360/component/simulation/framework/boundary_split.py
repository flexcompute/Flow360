"""
Boundary split infrastructure for handling surface/volume name mapping after meshing.

This module provides a unified framework for handling how boundaries (surfaces/volumes)
get renamed or split during the meshing process.

Naming convention:
- base_name: The name specified by user (e.g., "wing", "blade")
- full_name: The actual name in mesh (e.g., "fluid/wing", "rotatingBlock/blade__rotating_rotatingBlock")

Split scenarios:
- Zone prefix: "wing" -> "fluid/wing"
- Multi-zone split: "wing" -> "zone1/wing", "zone2/wing"
- RotationVolume: "blade" -> "rotatingBlock/blade__rotating_rotatingBlock"
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Set, runtime_checkable

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import (
    BOUNDARY_FULL_NAME_WHEN_NOT_FOUND,
    MirroredSurface,
    Surface,
    _SurfaceEntityBase,
)
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.log import log

if TYPE_CHECKING:
    from flow360.component.simulation.meshing_param.volume_params import RotationVolume
    from flow360.component.simulation.models.surface_models import Wall
    from flow360.component.simulation.simulation_params import SimulationParams


class SplitType(str, Enum):
    """Types of boundary splits that can occur during meshing."""

    ZONE_PREFIX = "zone_prefix"  # Normal zone/boundary prefix
    MULTI_ZONE = "multi_zone"  # Surface appears in multiple zones
    ROTATION_ENCLOSED = "rotation_enclosed"  # RotationVolume __rotating patch (enclosed)
    ROTATION_STATIONARY = "rotation_stationary"  # RotationVolume __rotating patch (stationary)


@dataclass
class BoundarySplitInfo:
    """Describes one variant of an original boundary after being split by the mesher."""

    full_name: str
    split_type: SplitType
    source_zone: Optional[str] = None


@runtime_checkable
class SplitProvider(Protocol):
    """
    Protocol for classes that can provide split information.

    Any class implementing this protocol can contribute split mappings
    to the BoundaryNameLookupTable.
    """

    def get_split_mappings(self, volume_mesh_meta_data: dict) -> Dict[str, List[BoundarySplitInfo]]:
        """Return split mappings for this provider."""

    def handled_by_provider(self, full_name: str) -> bool:
        """Check if boundary should be skipped by base lookup (handled by this provider)."""


# --- RotationVolume Split Provider ---
# Defined before BoundaryNameLookupTable since from_params() references it


class RotationVolumeSplitProvider:
    """
    SplitProvider implementation for RotationVolume __rotating patches.

    When a RotationVolume is defined, enclosed surfaces get renamed with __rotating suffix
    by the mesher. This provider detects those patterns and adds them to the lookup table.
    """

    def __init__(self, params: "SimulationParams"):
        self._params = params
        self._generated_full_names: Set[str] = set()

    # region ====private methods====

    def _get_rotation_volumes(self) -> List["RotationVolume"]:
        """Extract RotationVolume instances from params.meshing."""
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.meshing_param.params import (
            MeshingParams,
            ModularMeshingWorkflow,
        )
        from flow360.component.simulation.meshing_param.volume_params import (
            RotationVolume,
        )

        if self._params.meshing is None:
            return []

        volume_zones = None
        if isinstance(self._params.meshing, MeshingParams):
            volume_zones = self._params.meshing.volume_zones
        elif isinstance(self._params.meshing, ModularMeshingWorkflow):
            volume_zones = self._params.meshing.zones

        if volume_zones is None:
            return []

        return [zone for zone in volume_zones if isinstance(zone, RotationVolume)]

    def has_active_volumes(self) -> bool:
        """Check if there are any rotation volumes with enclosed entities."""
        rotation_volumes = self._get_rotation_volumes()
        for rotation_volume in rotation_volumes:
            if rotation_volume.enclosed_entities:
                return True
        return False

    @staticmethod
    def _find_zone_name(rotation_volume: "RotationVolume", zones: dict) -> Optional[str]:
        """Find the zone name for a rotation volume by matching entity name."""
        if not rotation_volume.entities or not rotation_volume.entities.stored_entities:
            return None

        entity_name = rotation_volume.entities.stored_entities[0].name

        if entity_name in zones:
            return entity_name

        matching_zone_names = [name for name in zones.keys() if entity_name in name]

        if len(matching_zone_names) == 1:
            return matching_zone_names[0]

        log.warning(
            f"[Internal] Ambiguous or no match found for rotation volume {rotation_volume.name}."
        )
        return None

    @staticmethod
    def _get_stationary_base_names(rotation_volume: "RotationVolume") -> Set[str]:
        """Collect base names of stationary enclosed entities."""
        stationary_base_names: Set[str] = set()
        if rotation_volume.stationary_enclosed_entities:
            for entity in rotation_volume.stationary_enclosed_entities.stored_entities:
                if isinstance(entity, (Surface, MirroredSurface)):
                    stationary_base_names.add(entity.name)
        return stationary_base_names

    def _add_enclosed_mappings(  # pylint: disable=too-many-arguments
        self,
        rotation_volume: "RotationVolume",
        zone_name: str,
        boundary_full_names: List[str],
        stationary_base_names: Set[str],
        mappings: Dict[str, List[BoundarySplitInfo]],
    ) -> None:
        """Add mappings for enclosed entities with __rotating patches."""
        if not rotation_volume.enclosed_entities:
            return

        for entity in rotation_volume.enclosed_entities.stored_entities:
            if not isinstance(entity, (Surface, MirroredSurface)):
                continue

            base_name = entity.name
            rotating_full_name = self._find_rotating_full_name(
                zone_name, base_name, boundary_full_names
            )
            if rotating_full_name is None:
                continue

            split_type = (
                SplitType.ROTATION_STATIONARY
                if base_name in stationary_base_names
                else SplitType.ROTATION_ENCLOSED
            )

            info = BoundarySplitInfo(
                full_name=rotating_full_name,
                split_type=split_type,
                source_zone=zone_name,
            )
            mappings.setdefault(base_name, []).append(info)

    @staticmethod
    def _find_rotating_full_name(
        zone_name: str, base_name: str, boundary_full_names: List[str]
    ) -> Optional[str]:
        """Find the __rotating boundary full_name matching the pattern."""
        rotating_pattern = (
            re.escape(zone_name)
            + r"/"
            + re.escape(base_name)
            + r"__rotating_"
            + re.escape(zone_name)
        )
        for full_name in boundary_full_names:
            if re.fullmatch(rotating_pattern, full_name):
                return full_name
        return None

    # endregion

    def get_split_mappings(self, volume_mesh_meta_data: dict) -> Dict[str, List[BoundarySplitInfo]]:
        """Return split mappings for RotationVolume __rotating patches."""
        mappings: Dict[str, List[BoundarySplitInfo]] = {}
        rotation_volumes = self._get_rotation_volumes()
        if not rotation_volumes:
            return mappings

        zones = volume_mesh_meta_data.get("zones", {})

        for rotation_volume in rotation_volumes:
            zone_name = self._find_zone_name(rotation_volume, zones)
            if zone_name is None:
                continue

            zone_meta = zones.get(zone_name)
            if zone_meta is None:
                continue

            boundary_full_names = zone_meta.get("boundaryNames", [])
            stationary_base_names = self._get_stationary_base_names(rotation_volume)
            self._add_enclosed_mappings(
                rotation_volume, zone_name, boundary_full_names, stationary_base_names, mappings
            )

        # Cache generated full names for handled_by_provider check
        for infos in mappings.values():
            for info in infos:
                self._generated_full_names.add(info.full_name)

        return mappings

    def handled_by_provider(self, full_name: str) -> bool:
        """
        Check if boundary is a __rotating patch (handled by this provider).
        Only returns True if the boundary was actually generated by this provider.
        """
        return full_name in self._generated_full_names


# --- Lookup Table ---


class BoundaryNameLookupTable:
    """
    Lookup table mapping base boundary names to their full names after meshing.

    This class is generic and extensible. Split logic is provided by:
    1. Built-in zone/boundary matching from metadata
    2. External SplitProvider implementations (e.g., RotationVolumeSplitProvider)

    Usage:
        # With SimulationParams (auto-creates RotationVolumeSplitProvider)
        lookup_table = BoundaryNameLookupTable.from_params(volume_mesh_meta_data, params)

        # With custom providers
        providers = [CustomSplitProvider()]
        lookup_table = BoundaryNameLookupTable(volume_mesh_meta_data, providers)
    """

    def __init__(
        self,
        volume_mesh_meta_data: dict,
        split_providers: Optional[List[SplitProvider]] = None,
    ):
        self._mapping: Dict[str, List[BoundarySplitInfo]] = {}
        self._providers = split_providers or []
        self._build_mapping(volume_mesh_meta_data)

    @classmethod
    def from_params(
        cls,
        volume_mesh_meta_data: dict,
        params: "SimulationParams",
    ) -> "BoundaryNameLookupTable":
        """
        Create a BoundaryNameLookupTable with default providers for SimulationParams.

        This factory method automatically creates the RotationVolumeSplitProvider
        based on the params.
        """
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.entity_info import VolumeMeshEntityInfo

        providers = []
        entity_info = params.private_attribute_asset_cache.project_entity_info

        # Skip if using existing volume mesh (split logic already handled in mesh)
        # Also skip if no active rotation volumes defined
        if not isinstance(entity_info, VolumeMeshEntityInfo):
            rotation_provider = RotationVolumeSplitProvider(params)
            if rotation_provider.has_active_volumes():
                providers.append(rotation_provider)

        return cls(volume_mesh_meta_data, providers)

    def _build_mapping(self, volume_mesh_meta_data: dict) -> None:
        """Build the complete mapping from base_name to split info."""
        self._add_zone_boundary_matches(volume_mesh_meta_data)

        for provider in self._providers:
            provider_mappings = provider.get_split_mappings(volume_mesh_meta_data)
            for base_name, infos in provider_mappings.items():
                for info in infos:
                    self._mapping.setdefault(base_name, []).append(info)

    def _add_zone_boundary_matches(self, volume_mesh_meta_data: dict) -> None:
        """Add mappings for standard zone/boundary name prefixing."""
        zones = volume_mesh_meta_data.get("zones", {}) or {}

        for zone_name, zone_meta in zones.items():
            boundary_names = zone_meta.get("boundaryNames", []) or []
            for full_name in boundary_names:
                # Check if any provider claims this boundary as special
                if any(p.handled_by_provider(full_name) for p in self._providers):
                    continue

                # Extract base_name from full_name
                if "/" in full_name:
                    parts = full_name.split("/", 1)
                    base_name = parts[1] if parts[0] == zone_name else full_name
                else:
                    base_name = full_name

                # Determine split type and add entry
                split_type = (
                    SplitType.MULTI_ZONE if base_name in self._mapping else SplitType.ZONE_PREFIX
                )
                info = BoundarySplitInfo(
                    full_name=full_name,
                    split_type=split_type,
                    source_zone=zone_name,
                )
                self._mapping.setdefault(base_name, []).append(info)

                # Also add full_name as key for passthrough lookup
                # (when entity.name is already a full_name, e.g., from volume mesh)
                if full_name != base_name:
                    self._mapping.setdefault(full_name, []).append(info)

    def get_split_info(self, base_name: str) -> List[BoundarySplitInfo]:
        """Get all split info for a base boundary name."""
        return self._mapping.get(base_name, [])


# --- Entity update functions (external entry point) ---


def update_entities_in_model(  # pylint: disable=too-many-branches
    model: Flow360BaseModel,
    lookup_table: BoundaryNameLookupTable,
    target_type: type = _SurfaceEntityBase,
) -> None:
    """
    Recursively (walk the params):
    1. update entity full_names in a model
    2. replace assignment with all the split entities
    using the lookup table.

    Handles:
    - Direct entity fields
    - EntityList fields
    - Nested Flow360BaseModel fields
    - Lists/tuples containing entities or models
    """
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.framework.param_utils import AssetCache

    for field in model.__dict__.values():
        if isinstance(field, AssetCache):
            continue

        if isinstance(field, target_type):
            _replace_with_actual_entities(field, lookup_table)

        elif isinstance(field, EntityList):
            added = []
            for entity in field.stored_entities:
                if isinstance(entity, target_type):
                    added.extend(_replace_with_actual_entities(entity, lookup_table))
            field.stored_entities.extend(added)

        elif isinstance(field, (list, tuple)):
            added = []
            for item in field:
                if isinstance(item, target_type):
                    added.extend(_replace_with_actual_entities(item, lookup_table))
                elif isinstance(item, Flow360BaseModel):
                    update_entities_in_model(item, lookup_table, target_type)
            if isinstance(field, list):
                field.extend(added)
            elif isinstance(field, tuple) and added:
                field += tuple(added)

        elif isinstance(field, Flow360BaseModel):
            update_entities_in_model(field, lookup_table, target_type)


def _replace_with_actual_entities(
    entity: _SurfaceEntityBase,
    lookup_table: BoundaryNameLookupTable,
) -> List[_SurfaceEntityBase]:
    """
    Update a single entity's full_name using the lookup table.

    Returns additional entities if the entity is split into multiple boundaries.
    """

    def _set_entity_full_name(entity: _SurfaceEntityBase, full_name: str) -> None:
        """Set the full_name on an entity."""
        with model_attribute_unlock(entity, "private_attribute_full_name"):
            entity.private_attribute_full_name = full_name

    def _create_entity_parts(
        original: _SurfaceEntityBase,
        split_infos: List[BoundarySplitInfo],
    ) -> List[_SurfaceEntityBase]:
        """Create entity copies for additional split results."""
        parts = []
        for info in split_infos:
            # Extract base name from full name (e.g., "zone/boundary" -> "boundary")
            parts.append(
                original.copy(
                    update={
                        "name": info.full_name,
                        "private_attribute_full_name": info.full_name,
                    }
                )
            )
        return parts

    split_infos = lookup_table.get_split_info(entity.name)

    if not split_infos:
        _set_entity_full_name(entity, BOUNDARY_FULL_NAME_WHEN_NOT_FOUND)
        return []

    _set_entity_full_name(entity, split_infos[0].full_name)
    return _create_entity_parts(entity, split_infos[1:])


# region --- Post-processing functions ---


def post_process_rotation_volume_entities(
    params: "SimulationParams",
    lookup_table: BoundaryNameLookupTable,
) -> None:
    """
    Filter RotationVolume enclosed_entities to only keep __rotating patches.

    After generic traversal, enclosed_entities may have been expanded to include
    both original and __rotating versions. This function filters to only keep
    the __rotating version for entities that have one.
    """
    provider = RotationVolumeSplitProvider(params)
    rotation_volumes = provider._get_rotation_volumes()  # pylint: disable=protected-access
    if not rotation_volumes:
        return

    both_types = (SplitType.ROTATION_ENCLOSED, SplitType.ROTATION_STATIONARY)
    stationary_only = (SplitType.ROTATION_STATIONARY,)

    for rotation_volume in rotation_volumes:
        _filter_entity_list_to_rotating(rotation_volume.enclosed_entities, lookup_table, both_types)
        _filter_entity_list_to_rotating(
            rotation_volume.stationary_enclosed_entities, lookup_table, stationary_only
        )


def _filter_entity_list_to_rotating(
    entity_list: Optional[EntityList],
    lookup_table: BoundaryNameLookupTable,
    target_split_types: tuple,
) -> None:
    """Filter an EntityList to only keep __rotating patches matching target split types."""
    if not entity_list or not entity_list.stored_entities:
        return

    # Collect matching __rotating full_names from the lookup table
    rotating_full_names = set()
    for split_infos in lookup_table._mapping.values():  # pylint: disable=protected-access
        for info in split_infos:
            if info.split_type in target_split_types:
                rotating_full_names.add(info.full_name)

    filtered_entities = []
    for entity in entity_list.stored_entities:
        if not isinstance(entity, (Surface, MirroredSurface)):
            filtered_entities.append(entity)
            continue

        # Keep if this entity is a __rotating version (by full_name match)
        if entity.full_name in rotating_full_names:
            filtered_entities.append(entity)
            continue

        # Skip if this entity has a __rotating counterpart (we'll keep the rotating one)
        split_infos = lookup_table.get_split_info(entity.name)
        has_rotating = any(info.split_type in target_split_types for info in split_infos)
        if not has_rotating:
            filtered_entities.append(entity)

    entity_list.stored_entities[:] = filtered_entities


def post_process_wall_models_for_rotating(
    params: "SimulationParams",
    lookup_table: BoundaryNameLookupTable,
) -> None:
    """
    Create Wall models for __rotating patches.

    For each Wall model that references a surface with a __rotating patch,
    create a new Wall model for the __rotating boundary. For stationary
    boundaries, set velocity=(0,0,0).
    """
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.models.surface_models import Wall

    if params.models is None:
        return

    # Build mapping of base_name -> (rotating_full_name, is_stationary)
    rotating_mappings: Dict[str, tuple] = {}
    for base_name, split_infos in lookup_table._mapping.items():  # pylint: disable=protected-access
        for info in split_infos:
            if info.split_type == SplitType.ROTATION_ENCLOSED:
                rotating_mappings[base_name] = (info.full_name, False)
            elif info.split_type == SplitType.ROTATION_STATIONARY:
                rotating_mappings[base_name] = (info.full_name, True)

    if not rotating_mappings:
        return

    models_to_add = []
    for model in params.models:
        if not isinstance(model, Wall):
            continue

        new_models = _create_rotating_wall_models(model, rotating_mappings)
        models_to_add.extend(new_models)

    if models_to_add:
        params.models.extend(models_to_add)


def _create_rotating_wall_models(
    wall_model: "Wall",
    rotating_mappings: Dict[str, tuple],
) -> List["Wall"]:
    """Create new Wall models for __rotating patches."""
    if not wall_model.entities or not wall_model.entities.stored_entities:
        return []

    stationary_surfaces = []
    non_stationary_surfaces = []

    for entity in wall_model.entities.stored_entities:
        if not isinstance(entity, (Surface, MirroredSurface)):
            continue

        base_name = entity.name
        if base_name not in rotating_mappings:
            continue

        rotating_full_name, is_stationary = rotating_mappings[base_name]
        rotating_base_name = (
            rotating_full_name.split("/")[-1] if "/" in rotating_full_name else rotating_full_name
        )
        rotating_entity = entity.copy(
            update={
                "name": rotating_base_name,
                "private_attribute_full_name": rotating_full_name,
            }
        )

        if is_stationary:
            stationary_surfaces.append(rotating_entity)
        else:
            non_stationary_surfaces.append(rotating_entity)

    models = []
    if stationary_surfaces:
        new_model = wall_model.copy(update={"velocity": ("0", "0", "0")})
        new_model.entities = stationary_surfaces
        models.append(new_model)

    if non_stationary_surfaces:
        new_model = wall_model.copy()
        new_model.entities = non_stationary_surfaces
        models.append(new_model)

    return models


# endregion
