"""Entity materialization utilities.

Provides stable keys, a default entity builder, and an in-place
materialization routine to convert entity dictionaries to shared
Pydantic model instances and perform per-list deduplication.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

import pydantic as pd

from flow360_schema.framework.entity.entity_materialization_context import (
    EntityMaterializationContext,
    get_entity_builder,
    get_entity_cache,
    get_entity_registry,
)
from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.framework.entity.entity_utils import (
    DEFAULT_NOT_MERGED_TYPES,
    deduplicate_entities,
    get_entity_key,
)
from flow360_schema.framework.validation.context import DeserializationContext
from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup, MirrorPlane
from flow360_schema.models.entities.output_entities import Point, PointArray, PointArray2D, Slice
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
    WindTunnelGhostSurface,
)
from flow360_schema.models.entities.volume_entities import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    GenericVolume,
    SeedpointVolume,
    Sphere,
    VoxelGrid,
)

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_registry import EntityRegistry


ENTITY_TYPE_MAP = {
    "Surface": Surface,
    "Edge": Edge,
    "GenericVolume": GenericVolume,
    "GeometryBodyGroup": GeometryBodyGroup,
    "CustomVolume": CustomVolume,
    "AxisymmetricBody": AxisymmetricBody,
    "Box": Box,
    "VoxelGrid": VoxelGrid,
    "Cylinder": Cylinder,
    "Sphere": Sphere,
    "ImportedSurface": ImportedSurface,
    "GhostSurface": GhostSurface,
    "GhostSphere": GhostSphere,
    "GhostCircularPlane": GhostCircularPlane,
    "Point": Point,
    "PointArray": PointArray,
    "PointArray2D": PointArray2D,
    "Slice": Slice,
    "SeedpointVolume": SeedpointVolume,
    "WindTunnelGhostSurface": WindTunnelGhostSurface,
    "MirroredSurface": MirroredSurface,
    "MirroredGeometryBodyGroup": MirroredGeometryBodyGroup,
    "MirrorPlane": MirrorPlane,
}


@cache
def _get_entity_type_adapter(cls: type[Any]) -> pd.TypeAdapter[Any]:
    """Cache entity TypeAdapters per concrete schema class."""
    return pd.TypeAdapter(cls)


def _build_entity_instance(entity_dict: dict[str, Any]) -> Any:
    """Construct a concrete entity instance from a dictionary via TypeAdapter."""
    type_name = entity_dict.get("private_attribute_entity_type_name")
    if not isinstance(type_name, str):
        raise ValueError("[Internal] Entity is missing a valid private_attribute_entity_type_name.")
    cls = ENTITY_TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(f"[Internal] Unknown entity type: {type_name}")
    with DeserializationContext():
        return _get_entity_type_adapter(cls).validate_python(entity_dict)


def _build_registry_index(registry: EntityRegistry) -> dict[tuple[str, str], Any]:
    """Build O(1) lookup index from EntityRegistry.

    Pre-builds a dict mapping (type_name, entity_id) -> entity for fast lookups.
    This converts O(n) registry.find_by_asset_id() to O(1) dict lookup.

    Parameters
    ----------
    registry : EntityRegistry
        Registry to index.

    Returns
    -------
    dict[tuple[str, str], Any]
        Index mapping (type_name, entity_id) to entity instances.
    """
    index = {}
    for entity_type, entities in registry.internal_registry.items():
        type_name = entity_type.__name__
        for entity in entities:
            entity_id = getattr(entity, "private_attribute_id", None)
            if entity_id:
                index[(type_name, entity_id)] = entity
    return index


def _lookup_entity_from_registry_cache(entity_dict: dict[str, Any], cache: dict[Any, Any]) -> Any:
    """Lookup entity from registry cache (Mode 2).

    Assumes cache has been pre-populated with registry index.

    Parameters
    ----------
    entity_dict : dict
        Entity dictionary with type_name and private_attribute_id.
    cache : dict
        Pre-populated cache with registry entities.

    Returns
    -------
    Any
        Entity instance from cache.

    Raises
    ------
    ValueError
        If entity missing ID, unknown type, or not found in cache.
    """
    type_name = entity_dict.get("private_attribute_entity_type_name")
    entity_id = entity_dict.get("private_attribute_id")

    if not entity_id:
        raise ValueError(
            f"[EntityMaterializer] Entity missing 'private_attribute_id' "
            f"when EntityRegistry is provided. Entity type: {type_name}, "
            f"name: {entity_dict.get('name', '<unknown>')}"
        )

    # O(1) lookup from pre-populated cache
    key = (type_name, entity_id)
    obj = cache.get(key)

    if obj is None:
        raise ValueError(
            f"[EntityMaterializer] Entity not found in EntityRegistry. "
            f"Type: {type_name}, ID: {entity_id}, "
            f"name: {entity_dict.get('name', '<unknown>')}"
        )

    return obj


def _build_entity_from_dict(
    entity_dict: dict[str, Any], cache: dict[Any, Any] | None, builder: Callable[..., Any]
) -> Any:
    """Build entity from dict with caching (Mode 1).

    Parameters
    ----------
    entity_dict : dict
        Entity dictionary to build from.
    cache : Optional[dict]
        Cache for reuse within params.
    builder : Callable
        Function to build entity instance.

    Returns
    -------
    Any
        Entity instance.
    """
    key = get_entity_key(entity_dict)
    obj = cache.get(key) if (cache and key in cache) else builder(entity_dict)
    if cache is not None and key not in cache:
        cache[key] = obj
    return obj


def _convert_entity_dict_to_object(
    item: dict[str, Any],
    is_registry_mode: bool,
    cache: dict[Any, Any] | None,
    builder: Callable[..., Any] | None,
) -> tuple[Any, tuple[Any, ...]]:
    """Convert entity dict to object instance.

    Parameters
    ----------
    item : dict
        Entity dictionary.
    is_registry_mode : bool
        True if using EntityRegistry mode (Mode 2).
    cache : Optional[dict]
        Cache for performance (pre-populated in Mode 2).  Else a progressive built cache.
    builder : Optional[Callable]
        Builder for Mode 1.

    Returns
    -------
    tuple[Any, tuple]
        (entity_object, key) tuple.
    """
    if is_registry_mode:
        # Mode 2: Lookup from pre-populated cache (O(1))
        obj = _lookup_entity_from_registry_cache(item, cache)  # type: ignore[arg-type]
        type_name = item.get("private_attribute_entity_type_name")
        entity_id = item.get("private_attribute_id")
        key = (type_name, entity_id)
    else:
        # Mode 1: Create and cache within params
        obj = _build_entity_from_dict(item, cache, builder)  # type: ignore[arg-type]
        key = get_entity_key(item)

    return obj, key


def _deserialize_used_selectors_and_build_lookup(params_as_dict: dict[str, Any]) -> dict[str, EntitySelector]:
    """Deserialize asset_cache.used_selectors in-place and build selector_id -> selector lookup."""
    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    if not isinstance(asset_cache, dict):
        return {}

    raw_used_selectors = asset_cache.get("used_selectors")
    if not isinstance(raw_used_selectors, list) or not raw_used_selectors:
        return {}

    try:
        selector_list = pd.TypeAdapter(list[EntitySelector]).validate_python(raw_used_selectors)
    except pd.ValidationError as e:
        # Prepend the correct path to error locations so they match SimulationParams structure
        errors_with_path = []
        for err in e.errors():
            new_loc = ("private_attribute_asset_cache", "used_selectors") + tuple(err["loc"])
            errors_with_path.append({**err, "loc": new_loc})
        raise pd.ValidationError.from_exception_data(
            title=e.title,
            line_errors=errors_with_path,  # type: ignore[arg-type]
        ) from None

    selector_lookup = {selector.selector_id: selector for selector in selector_list}

    # Keep used_selectors as a list, but ensure it contains deserialized EntitySelector instances.
    asset_cache["used_selectors"] = selector_list
    return selector_lookup


def _materialize_stored_entities_list_in_node(
    node: dict[str, Any],
    *,
    is_registry_mode: bool,
    cache: dict[Any, Any] | None,
    builder: Callable[..., Any] | None,
    not_merged_types: set[str],
) -> None:
    """Materialize node['stored_entities'] in-place if present."""
    stored_entities = node.get("stored_entities")
    if not isinstance(stored_entities, list):
        return

    def processor(item: Any) -> tuple[Any, tuple[Any, ...]]:
        if isinstance(item, dict):
            return _convert_entity_dict_to_object(item, is_registry_mode, cache, builder)
        # Already materialized
        return item, get_entity_key(item)

    node["stored_entities"] = deduplicate_entities(
        stored_entities,
        processor=processor,
        not_merged_types=not_merged_types,  # type: ignore[arg-type]
    )


def _materialize_selectors_list_in_node(node: dict[str, Any], selector_lookup: dict[str, EntitySelector]) -> None:
    """Replace selector tokens in node['selectors'] with shared EntitySelector instances."""
    selectors = node.get("selectors")
    if not isinstance(selectors, list) or not selectors:
        return

    materialized_selectors: list[Any] = []
    for selector_item in selectors:
        if isinstance(selector_item, str):
            # ==== Selector token (str) ====
            selector_object = selector_lookup.get(selector_item)
            if selector_object is None:
                raise ValueError(
                    "[Internal] Selector token not found in "
                    "private_attribute_asset_cache.used_selectors: "
                    f"{selector_item}"
                )
            materialized_selectors.append(selector_object)
        elif isinstance(selector_item, dict):
            # ==== Inline selector definition (dict, pre-submit JSON) ====
            # Cloud/Production JSON data will only contain selector tokens (str).
            # Local pre-upload JSON (from model_dump) will contain inline selector definitions (dict).
            # At local validation, `selector_lookup` is empty.
            # Since it is presubmit, no need to "materialize", "deserialize" is fine.
            try:
                materialized_selectors.append(EntitySelector.deserialize(selector_item))
            except pd.ValidationError:
                # Keep the invalid dict as-is, let SimulationParams.model_validate handle the error.
                # This preserves the full error location path (e.g., "models.0.entities.selectors.0.children...")
                # instead of a truncated path (e.g., "children...").
                materialized_selectors.append(selector_item)
        elif isinstance(selector_item, EntitySelector):
            # ==== Already materialized EntitySelector ====
            # When materialize_entities_and_selectors_in_place is called multiple times
            # on the same params dict (e.g., repeated validation or upload after preprocessing),
            # selectors may already be EntitySelector objects. Pass through unchanged.
            materialized_selectors.append(selector_item)
        else:
            raise TypeError(
                "[Internal] Unsupported selector item type in selectors list. "
                "Expected selector tokens (str/dict) or EntitySelector instances. Got: "
                f"{type(selector_item)}"
            )
    node["selectors"] = materialized_selectors


def materialize_entities_and_selectors_in_place(
    params_as_dict: dict[str, Any],
    *,
    entity_builder: Callable[[dict[str, Any]], Any] | None = None,
    not_merged_types: set[str] = DEFAULT_NOT_MERGED_TYPES,  # type: ignore[assignment]
    entity_registry: EntityRegistry | None = None,
) -> dict[str, Any]:
    """
    From raw dict simulation params:
    1. Materialize `stored_entities` dicts to shared instances and dedupe per list in-place.
    2. Materialize `selectors` list to shared EntitySelector instances.

    Two operation modes:

    Mode 1 (entity_registry=None): Intra-params deduplication
        - Converts dict entries to instances using a scoped cache for reuse
        - Entities appearing multiple times within params share the same object
        - Deduplicates within each stored_entities list

    Mode 2 (entity_registry provided): Registry reference mode
        - ALL entities MUST already exist in the EntityRegistry
        - Replaces entity dicts with references to registry instances
        - Errors if an entity is not found in the registry
        - No new entity instances are created

        When called by validate_model(), the entity_registry can be provided by ParamsValidationInfo.
        BLOCKED: This require all entities to have private_attribute_id set
                 Which due to legacy reasons is not the case for all entities.

    Parameters
    ----------
    params_as_dict : dict
        The simulation params dictionary to materialize in-place.
    entity_builder : Optional[Callable[[dict], Any]]
        Function to build entity instances from dicts. When omitted, the
        built-in schema entity builder is used.
    not_merged_types : set[str]
        Entity types to skip deduplication (e.g., Point).
    entity_registry : Optional[EntityRegistry]
        EntityRegistry containing canonical entity instances.
        When provided, all entities must exist in registry (Mode 2).
    """

    selector_lookup = _deserialize_used_selectors_and_build_lookup(params_as_dict)
    entity_builder = entity_builder or _build_entity_instance

    with EntityMaterializationContext(builder=entity_builder, entity_registry=entity_registry):
        # Pre-build registry index for O(1) lookups (Mode 2 only)
        registry = get_entity_registry()
        cache = get_entity_cache()
        builder = get_entity_builder()

        is_registry_mode = registry is not None
        if is_registry_mode:
            # Mode 2: Pre-populate cache with registry index for O(1) lookups
            registry_index = _build_registry_index(registry)  # type: ignore[arg-type]
            cache.update(registry_index)  # type: ignore[union-attr]

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                _materialize_stored_entities_list_in_node(
                    node,
                    is_registry_mode=is_registry_mode,
                    cache=cache,
                    builder=builder,
                    not_merged_types=not_merged_types,
                )
                _materialize_selectors_list_in_node(node, selector_lookup)

                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for it in node:
                    visit(it)

        visit(params_as_dict)

    return params_as_dict


__all__ = [
    "ENTITY_TYPE_MAP",
    "materialize_entities_and_selectors_in_place",
]
