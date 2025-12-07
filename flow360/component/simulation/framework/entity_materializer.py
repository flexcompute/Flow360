"""Entity materialization utilities.

Provides mapping from entity type names to classes, stable keys, and an
in-place materialization routine to convert entity dictionaries to shared
Pydantic model instances and perform per-list deduplication.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import pydantic as pd

from flow360.component.simulation.framework.entity_materialization_context import (
    EntityMaterializationContext,
    get_entity_builder,
    get_entity_cache,
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

    Prefer (type, private_attribute_id); if missing, hash a sanitized
    JSON-dumped copy (excluding volatile fields like private_attribute_input_cache).
    """
    t = d.get("private_attribute_entity_type_name")
    pid = d.get("private_attribute_id")
    if pid:
        return (t, pid)
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


def materialize_entities_in_place(
    params_as_dict: dict,
    *,
    not_merged_types: set[str] = frozenset({"Point"}),
    # TODO: Does this have all entities including the draft entities?
    entity_pool: Optional[dict] = None,
) -> dict:
    """Materialize entity dicts to shared instances and dedupe per list in-place.

    - Converts dict entries to instances using a scoped cache for reuse.
    - Deduplicates within each stored_entities list; skips types in not_merged_types.
    - If called re-entrantly on an already materialized structure, object
      instances are passed through and participate in per-list deduplication.

    Parameters
    ----------
    params_as_dict : dict
        The simulation params dictionary to materialize in-place.
    not_merged_types : set[str]
        Entity types to skip deduplication (e.g., Point).
    entity_pool : Optional[dict]
        Pre-existing entity instances keyed by (type_name, private_attribute_id).
        When provided, entities matching these keys will reuse the pool instances
        instead of creating new ones. This enables reference identity between
        entity_info and params.
    """

    def visit(node):
        if isinstance(node, dict):
            stored_entities = node.get("stored_entities", None)
            if isinstance(stored_entities, list):
                cache = get_entity_cache()
                builder = get_entity_builder()
                new_list = []
                seen = set()
                for item in stored_entities:
                    if isinstance(item, dict):
                        key = _stable_entity_key_from_dict(item)
                        obj = cache.get(key) if (cache and key in cache) else builder(item)
                        if cache is not None and key not in cache:
                            cache[key] = obj
                    else:
                        # If already materialized (e.g., re-entrant call), passthrough
                        obj = item
                        key = _stable_entity_key_from_dict(
                            {
                                "private_attribute_entity_type_name": getattr(
                                    obj, "private_attribute_entity_type_name", type(obj).__name__
                                ),
                                "private_attribute_id": getattr(obj, "private_attribute_id", None),
                                "name": getattr(obj, "name", None),
                            }
                        )
                    entity_type = getattr(obj, "private_attribute_entity_type_name", None)
                    if entity_type in not_merged_types:
                        new_list.append(obj)
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    new_list.append(obj)
                node["stored_entities"] = new_list
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for it in node:
                visit(it)

    with EntityMaterializationContext(builder=_build_entity_instance, entity_pool=entity_pool):
        visit(params_as_dict)
    return params_as_dict
