"""Entity materialization utilities.

Provides mapping from entity type names to classes, stable keys, and an
in-place materialization routine to convert entity dictionaries to shared
Pydantic model instances and perform per-list deduplication.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Optional

import pydantic as pd

from flow360.component.simulation.framework.entity_materialization_context import (
    EntityMaterializationContext,
    get_entity_builder,
    get_entity_cache,
    get_entity_registry,
)
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
    GenericVolume,
    GeometryBodyGroup,
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    ImportedSurface,
    SeedpointVolume,
    SnappyBody,
    Surface,
    WindTunnelGhostSurface,
)

if TYPE_CHECKING:
    from flow360.component.simulation.framework.entity_registry import EntityRegistry

ENTITY_TYPE_MAP = {
    "Surface": Surface,
    "Edge": Edge,
    "GenericVolume": GenericVolume,
    "GeometryBodyGroup": GeometryBodyGroup,
    "CustomVolume": CustomVolume,
    "Box": Box,
    "Cylinder": Cylinder,
    "ImportedSurface": ImportedSurface,
    "GhostSurface": GhostSurface,
    "GhostSphere": GhostSphere,
    "GhostCircularPlane": GhostCircularPlane,
    "Point": Point,
    "PointArray": PointArray,
    "PointArray2D": PointArray2D,
    "Slice": Slice,
    "SeedpointVolume": SeedpointVolume,
    "SnappyBody": SnappyBody,
    "WindTunnelGhostSurface": WindTunnelGhostSurface,
}


def _stable_entity_key_from_dict(d: dict) -> tuple:
    """Return a stable deduplication key for an entity dict.

    Prefer (type, private_attribute_id);
    if missing, hash a sanitized JSON-dumped copy (excluding volatile fields like private_attribute_input_cache).
    """
    t = d.get("private_attribute_entity_type_name")
    pid = d.get("private_attribute_id")
    if pid:
        return (t, pid)
    # Fallback mode, possibly because:
    # 1. Test does not set private_attribute_id.
    # 2. Very legacy JSON where private_attribute_id was not even defined back then.
    # 3. Ghost entities (we should have enforced private_attribute_id required for all entities)
    # All production persistent data should have private_attribute_id set.
    data = {k: v for k, v in d.items() if k not in ("private_attribute_input_cache",)}
    return (t, hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest())


def _stable_entity_key_from_obj(o: Any) -> tuple:
    """Return a stable deduplication key for an entity object instance."""
    t = getattr(o, "private_attribute_entity_type_name", type(o).__name__)
    pid = getattr(o, "private_attribute_id", None)
    return (t, pid) if pid else (t, id(o))


def _build_entity_instance(entity_dict: dict):
    """Construct a concrete entity instance from a dictionary via TypeAdapter."""
    type_name = entity_dict.get("private_attribute_entity_type_name")
    cls = ENTITY_TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(f"[Internal] Unknown entity type: {type_name}")
    return pd.TypeAdapter(cls).validate_python(entity_dict)


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


def _lookup_entity_from_registry_cache(entity_dict: dict, cache: dict) -> Any:
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


def _build_entity_from_dict(entity_dict: dict, cache: Optional[dict], builder: Callable) -> Any:
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
    key = _stable_entity_key_from_dict(entity_dict)
    obj = cache.get(key) if (cache and key in cache) else builder(entity_dict)
    if cache is not None and key not in cache:
        cache[key] = obj
    return obj


def _convert_entity_dict_to_object(
    item: dict,
    is_registry_mode: bool,
    cache: Optional[dict],
    builder: Optional[Callable],
) -> tuple[Any, tuple]:
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
        obj = _lookup_entity_from_registry_cache(item, cache)
        type_name = item.get("private_attribute_entity_type_name")
        entity_id = item.get("private_attribute_id")
        key = (type_name, entity_id)
    else:
        # Mode 1: Create and cache within params
        obj = _build_entity_from_dict(item, cache, builder)
        key = _stable_entity_key_from_dict(item)

    return obj, key


def _get_entity_key_from_object(obj: Any) -> tuple:
    """Extract stable key from already-materialized entity object.

    Parameters
    ----------
    obj : Any
        Entity object instance.

    Returns
    -------
    tuple
        Stable key (type_name, id).
    """
    # Optimized: direct key extraction without temporary dict
    entity_type = getattr(obj, "private_attribute_entity_type_name", type(obj).__name__)
    entity_id = getattr(obj, "private_attribute_id", None)

    if entity_id:
        return (entity_type, entity_id)

    # Fallback: use object identity if no ID (rare case)
    return (entity_type, id(obj))


def _process_stored_entities_list(
    stored_entities: list,
    not_merged_types: set[str],
    is_registry_mode: bool,
    cache: Optional[dict],
    builder: Optional[Callable],
) -> list:
    """Process stored_entities list: materialize and deduplicate.

    Parameters
    ----------
    stored_entities : list
        List of entity dicts or objects.
    not_merged_types : set[str]
        Entity types to skip deduplication.
    is_registry_mode : bool
        True if using EntityRegistry mode.
    cache : Optional[dict]
        Cache for entity lookups/storage.
    builder : Optional[Callable]
        Builder function for Mode 1.

    Returns
    -------
    list
        Processed list with materialized and deduplicated entities.
    """
    new_list = []
    seen = set()

    for item in stored_entities:
        # Convert to object instance
        if isinstance(item, dict):
            obj, key = _convert_entity_dict_to_object(item, is_registry_mode, cache, builder)
        else:
            # Already materialized (re-entrant call), passthrough
            obj = item
            key = _get_entity_key_from_object(obj)

        # Apply deduplication logic
        entity_type = getattr(obj, "private_attribute_entity_type_name", None)
        if entity_type in not_merged_types:
            # Skip deduplication for these types
            new_list.append(obj)
            continue

        if key in seen:
            # Duplicate, skip
            continue

        seen.add(key)
        new_list.append(obj)

    return new_list


def materialize_entities_in_place(
    params_as_dict: dict,
    *,
    not_merged_types: set[str] = frozenset({"Point"}),
    entity_registry: Optional[EntityRegistry] = None,
) -> dict:
    """Materialize entity dicts to shared instances and dedupe per list in-place.

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
        # NOTE: Possibly unnecessary:
        This function as of now is mostly used by the validate_model().
        The slot for entity_registry was reserved for the entity registry coming from draft context.
        However the only scenario where validate_model() will be called within the draft context,
        is when the user eventually submits the simulation json.
        Due to set_up_params_for_uploading(), The entity_info before validation has already been replaced by the
        entity info from the draft context. So enforcing the entity registry here does not provide any benefit.
        Because at this point user has finished all the editing and it's not beneficial to ensure the entity
        deserialized by validate_model() has to be the same object as the one in draft context entity registry.

    Parameters
    ----------
    params_as_dict : dict
        The simulation params dictionary to materialize in-place.
    not_merged_types : set[str]
        Entity types to skip deduplication (e.g., Point).
    entity_registry : Optional[EntityRegistry]
        EntityRegistry containing canonical entity instances.
        When provided, all entities must exist in registry (Mode 2).
    """

    with EntityMaterializationContext(
        builder=_build_entity_instance, entity_registry=entity_registry
    ):
        # Pre-build registry index for O(1) lookups (Mode 2 only)
        registry = get_entity_registry()
        cache = get_entity_cache()
        builder = get_entity_builder()

        is_registry_mode = registry is not None
        if is_registry_mode:
            # Mode 2: Pre-populate cache with registry index for O(1) lookups
            registry_index = _build_registry_index(registry)
            cache.update(registry_index)

        def visit(node):
            if isinstance(node, dict):
                stored_entities = node.get("stored_entities", None)
                if isinstance(stored_entities, list):
                    node["stored_entities"] = _process_stored_entities_list(
                        stored_entities, not_merged_types, is_registry_mode, cache, builder
                    )
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for it in node:
                    visit(it)

    with EntityMaterializationContext(builder=_build_entity_instance):
        visit(params_as_dict)

    return params_as_dict
