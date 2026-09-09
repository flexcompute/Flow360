"""Entity materialization utilities.

Provides stable keys, a default entity builder, and an in-place
materialization routine to convert entity dictionaries to shared
Pydantic model instances and perform per-list deduplication.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import pydantic as pd

from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.framework.entity.entity_utils import (
    DEFAULT_NOT_MERGED_TYPES,
    deduplicate_entities,
    get_entity_key,
    is_entity_reference_group,
)
from flow360_schema.framework.validation.context import DeserializationContext
from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup, MirrorPlane
from flow360_schema.models.entities.output_entities import Point, PointArray, PointArray2D, Slice
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
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


def _resolve_reference_group(
    group: dict[str, Any],
    *,
    reference_index: dict[tuple[str, str], Any],
    has_registry: bool,
    path: str,
) -> list[Any]:
    """Expand a compact reference group {"type": ..., "ids": [...]} into entity instances.

    Every reference — ghost entities included — must resolve against the registry index.
    """
    type_name = group["type"]
    ids = group["ids"]
    if not isinstance(type_name, str) or not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError(f"[EntityMaterializer] Malformed entity reference group at '{path}.stored_entities': {group}")

    resolved = []
    for entity_id in ids:
        obj = reference_index.get((type_name, entity_id))
        if obj is None:
            if not has_registry:
                raise ValueError(
                    f"[EntityMaterializer] Compact entity reference (type: {type_name}, id: {entity_id}) "
                    f"at '{path}.stored_entities' requires an EntityRegistry backed by "
                    "private_attribute_asset_cache.project_entity_info, but none is available."
                )
            raise ValueError(
                f"[EntityMaterializer] Entity reference not found in EntityRegistry. "
                f"Type: {type_name}, ID: {entity_id}, at: '{path}.stored_entities'"
            )
        resolved.append(obj)
    return resolved


def _materialize_stored_entities_list_in_node(
    node: dict[str, Any],
    *,
    reference_index: dict[tuple[str, str], Any],
    has_registry: bool,
    not_merged_types: set[str],
    path: str,
    groups_only: bool = False,
) -> None:
    """Materialize node['stored_entities'] in-place if present."""
    stored_entities = node.get("stored_entities")
    if not isinstance(stored_entities, list):
        return

    flattened: list[Any] = []
    for item in stored_entities:
        if is_entity_reference_group(item):
            flattened.extend(
                _resolve_reference_group(item, reference_index=reference_index, has_registry=has_registry, path=path)
            )
        else:
            flattened.append(item)

    if groups_only:
        # Inside private_attribute_input_cache: only compact reference groups are
        # resolved (the recorded constructor needs real entities); inline payloads
        # stay untouched so the constructor validates them itself, preserving the
        # legacy error collection and locations. Disposable with PR #6080.
        node["stored_entities"] = flattened
        return

    def check_item(index: int, item: Any) -> Any:
        if not isinstance(item, dict) or "private_attribute_entity_type_name" not in item:
            # Already-materialized instance, or multi-constructor SHORTHAND
            # ({"type_name": ..., "private_attribute_input_cache": ...}) — a constructor
            # payload, not an entity definition; it passes through verbatim for
            # parse_model_dict, which runs after materialization.
            return item
        # The wire is compact-only: definitions live in project_entity_info and
        # assignments are reference groups. An inline entity definition means the
        # data predates the updater milestone or bypassed serialization entirely.
        raise ValueError(
            f"[EntityMaterializer] Inline entity definition at "
            f"'{path}.stored_entities[{index}]' (type: {item.get('private_attribute_entity_type_name')}, "
            f"name: {item.get('name')}) — the wire format carries only "
            '{"type", "ids"} reference groups; definitions belong in '
            "private_attribute_asset_cache.project_entity_info."
        )

    built = [check_item(index, item) for index, item in enumerate(flattened)]

    def processor(item: Any) -> tuple[Any, tuple[Any, ...]]:
        if isinstance(item, dict):
            # Shorthand dicts never merge.
            return item, ("__multi_constructor_shorthand__", id(item))
        return item, get_entity_key(item)

    node["stored_entities"] = deduplicate_entities(
        built,
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
    not_merged_types: set[str] = DEFAULT_NOT_MERGED_TYPES,  # type: ignore[assignment]
    entity_registry: EntityRegistry | None = None,
) -> dict[str, Any]:
    """
    From raw dict simulation params:
    1. Expand `stored_entities` reference groups to registry instances, dedupe per list in-place.
    2. Materialize `selectors` list to shared EntitySelector instances.

    Compact reference groups ({"type": ..., "ids": [...]}) expand in assignment
    order to instances of the provided ``entity_registry``; a reference missing
    from the registry is a hard error, and so is an inline entity definition
    (the wire is compact-only — definitions live in project_entity_info).
    #TOAI: "already-materialized instances", when would we have this situation?
    Multi-constructor shorthand payloads and already-materialized instances pass
    through untouched.

    Parameters
    ----------
    params_as_dict : dict
        The simulation params dictionary to materialize in-place.
    not_merged_types : set[str]
        Entity types to skip deduplication (e.g., Point).
    entity_registry : Optional[EntityRegistry]
        EntityRegistry containing canonical entity instances that compact
        reference groups resolve to. Optional because registry-less callers are
        legitimate: params without an asset cache / entity_info (locally
        constructed files, defaults, fragments) and
        ``expand_entity_list_in_context`` materializing leftover items in
        already-validated lists (client results post-processing, solver
        translator). Reference groups in a registry-less call are a hard error.
    """

    selector_lookup = _deserialize_used_selectors_and_build_lookup(params_as_dict)

    has_registry = entity_registry is not None
    # (type_name, entity_id) -> instance index resolving compact reference groups.
    reference_index = _build_registry_index(entity_registry) if has_registry else {}  # type: ignore[arg-type]

    def visit(node: Any, path: str, inside_input_cache: bool = False) -> None:
        if isinstance(node, dict):
            _materialize_stored_entities_list_in_node(
                node,
                reference_index=reference_index,
                has_registry=has_registry,
                not_merged_types=not_merged_types,
                path=path,
                groups_only=inside_input_cache,
            )
            _materialize_selectors_list_in_node(node, selector_lookup)

            for key, value in node.items():
                if not path and key == "private_attribute_asset_cache":
                    # The asset cache is the definitions store: entity_info is
                    # deserialized separately (with its own bounding-reference
                    # expansion) and used_selectors was handled above.
                    continue
                visit(
                    value,
                    f"{path}.{key}" if path else str(key),
                    inside_input_cache or key == "private_attribute_input_cache",
                )
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]", inside_input_cache)

    visit(params_as_dict, "")

    return params_as_dict


__all__ = [
    "ENTITY_TYPE_MAP",
    "materialize_entities_and_selectors_in_place",
]
