"""Entity expansion utilities that depend only on schema-owned types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)
from flow360_schema.framework.entity.entity_utils import (
    walk_object_tree_with_cycle_detection,
)

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_base import EntityBase
    from flow360_schema.framework.entity.entity_list import EntityList
    from flow360_schema.framework.entity.entity_registry import EntityRegistry


def _get_mapping_or_attribute(obj: Any, name: str) -> Any:
    """Return a value from either a dict-like payload or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _register_mirror_entities_in_registry(registry: EntityRegistry, mirror_status: Any) -> None:
    """Register mirror-related entities (planes + derived mirrored entities) into registry.

    This helper is shared by both dict-based and params-based registry builders to ensure
    consistent selector expansion coverage.
    """
    if not mirror_status:
        return

    # Lazy import to avoid pulling these models unless mirror status is actually present.
    from flow360_schema.models.asset_cache import MirrorStatus
    from flow360_schema.models.entities.geometry_entities import MirrorPlane

    # Dict path: deserialize to MirrorStatus
    if isinstance(mirror_status, dict):
        mirror_status = MirrorStatus.deserialize(mirror_status)

    # Object path: MirrorStatus (or compatible) with is_empty()
    if hasattr(mirror_status, "is_empty") and mirror_status.is_empty():
        return

    for plane in getattr(mirror_status, "mirror_planes", []) or []:
        if isinstance(plane, MirrorPlane):
            registry.register(plane)
    for mirrored_group in getattr(mirror_status, "mirrored_geometry_body_groups", []) or []:
        registry.register(mirrored_group)
    for mirrored_surface in getattr(mirror_status, "mirrored_surfaces", []) or []:
        registry.register(mirrored_surface)


def get_entity_info_and_registry_from_asset_cache(asset_cache: Any) -> tuple[Any, Any]:
    """
    Create EntityInfo and EntityRegistry from an asset cache object or dict.

    The EntityInfo owns the entities, and EntityRegistry holds references to them.
    Callers must keep entity_info alive as long as registry is used.
    """
    from flow360_schema.framework.entity.entity_registry import EntityRegistry
    from flow360_schema.models.entity_info import parse_entity_info_model

    if asset_cache is None:
        raise ValueError("[Internal] asset_cache is required to build entity registry.")

    entity_info = _get_mapping_or_attribute(asset_cache, "project_entity_info")
    if entity_info is None:
        raise ValueError("[Internal] project_entity_info not found in asset cache.")

    if isinstance(entity_info, dict):
        entity_info = parse_entity_info_model(entity_info)

    registry = EntityRegistry.from_entity_info(entity_info)

    mirror_status = _get_mapping_or_attribute(asset_cache, "mirror_status")
    _register_mirror_entities_in_registry(registry, mirror_status)

    return entity_info, registry


def get_entity_info_and_registry_from_dict(params_as_dict: dict[str, Any]) -> tuple[Any, Any]:
    """
    Create EntityInfo and EntityRegistry from simulation params dictionary.

    The EntityInfo owns the entities, and EntityRegistry holds references to them.
    Callers must keep entity_info alive as long as registry is used.

    Parameters
    ----------
    params_as_dict : dict
        Simulation parameters as dictionary containing private_attribute_asset_cache.

    Returns
    -------
    tuple[EntityInfo, EntityRegistry]
        (entity_info, registry) where entity_info owns entities and registry references them.
    """
    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    if asset_cache is None:
        raise ValueError("[Internal] private_attribute_asset_cache not found in params_as_dict.")
    return get_entity_info_and_registry_from_asset_cache(asset_cache)


def get_registry_from_asset_cache(asset_cache: Any) -> Any:
    """Create an EntityRegistry from an asset cache object or dict."""
    return get_entity_info_and_registry_from_asset_cache(asset_cache)[1]


def expand_entity_list_with_registry(
    entity_list: EntityList,
    registry: Any | None = None,
    *,
    return_names: bool = False,
) -> list[EntityBase] | list[str]:
    """
    Expand selectors for a deserialized EntityList within an EntityRegistry context.

    When no selectors are present, `registry` may be omitted and explicit stored entities
    will still be materialized and returned as-is. Explicit entities are expected to
    have already been filtered by EntityList during validation.
    """
    stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
    selectors = list(getattr(entity_list, "selectors", []) or [])

    if selectors:
        if registry is None:
            raise Flow360ValueError("An EntityRegistry is required to expand selectors in the given EntityList.")

        from flow360_schema.framework.entity.entity_selector import (
            resolve_entity_list_selectors,
        )

        try:
            stored_entities = resolve_entity_list_selectors(
                registry,
                entity_list,
                selector_cache={},
                merge_mode="merge",
            )
        except ValueError as exc:
            raise Flow360ValueError(
                "Failed to find any valid entities in the input. "
                "Has the simulationParams been manually edited since loading from the cloud "
                "or have you changed the cloud resource for which the SimulationParams is being used?"
            ) from exc

    if not stored_entities:
        return []

    if not all(hasattr(entity, "name") for entity in stored_entities):
        wrapper = {"stored_entities": stored_entities}
        materialize_entities_and_selectors_in_place(wrapper)
        stored_entities = wrapper.get("stored_entities", [])

    if return_names:
        return [entity.name for entity in stored_entities]
    return stored_entities


def expand_all_entity_lists_with_registry_in_place(
    root_obj: Any,
    *,
    registry: Any,
    merge_mode: Literal["merge", "replace"] = "merge",
    expansion_map: dict[str, list[str]] | None = None,
) -> None:
    """Resolve selectors for all EntityList objects under `root_obj` in-place."""
    from flow360_schema.framework.entity.entity_list import EntityList
    from flow360_schema.framework.entity.entity_selector import (
        resolve_entity_list_selectors,
    )

    selector_cache: dict[str, Any] = {}

    def _process_entity_list(obj: Any) -> bool:
        if isinstance(obj, EntityList):
            resolved_entities = resolve_entity_list_selectors(
                registry,
                obj,
                selector_cache=selector_cache,
                merge_mode=merge_mode,
                expansion_map=expansion_map,
            )
            obj.stored_entities = resolved_entities
            return False
        return True

    walk_object_tree_with_cycle_detection(root_obj, _process_entity_list, check_dict=True)


__all__ = [
    "_register_mirror_entities_in_registry",
    "expand_all_entity_lists_with_registry_in_place",
    "expand_entity_list_with_registry",
    "get_entity_info_and_registry_from_asset_cache",
    "get_entity_info_and_registry_from_dict",
    "get_registry_from_asset_cache",
]
