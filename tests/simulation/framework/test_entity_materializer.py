import copy
from typing import Optional

import pydantic as pd
import pytest

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_in_place,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.primitives import Surface


def _mk_entity(name: str, entity_type: str, eid: Optional[str] = None) -> dict:
    d = {"name": name, "private_attribute_entity_type_name": entity_type}
    if eid is not None:
        d["private_attribute_id"] = eid
    return d


def _mk_surface_dict(name: str, eid: str):
    return {
        "private_attribute_entity_type_name": "Surface",
        "private_attribute_id": eid,
        "name": name,
    }


def _mk_point_dict(name: str, eid: str, coords=(0.0, 0.0, 0.0)):
    return {
        "private_attribute_entity_type_name": "Point",
        "private_attribute_id": eid,
        "name": name,
        "location": {"units": "m", "value": list(coords)},
    }


def test_materializes_dicts_and_shares_instances_across_lists():
    """
    Test: Entity materializer converts dicts to Pydantic instances and shares them globally.

    Purpose:
    - Verify that materialize_entities_in_place() converts entity dicts to model instances
    - Verify that entities with same (type, id) are the same Python object (by identity)
    - Verify that instance sharing works across different nodes in the params tree
    - Verify that materialization is idempotent with respect to instance identity

    Expected behavior:
    - Input: Entity dicts with same IDs in different locations (nodes a and b)
    - Process: Materialization uses global cache keyed by (type, id)
    - Output: Same instances appear in both locations (a_list[0] is b_list[1])

    This enables memory efficiency and supports identity-based entity comparison.
    """
    params = {
        "a": {
            "stored_entities": [
                _mk_entity("wing", "Surface", eid="s-1"),
                _mk_entity("tail", "Surface", eid="s-2"),
            ]
        },
        "b": {
            "stored_entities": [
                # same ids as in node a
                _mk_entity("tail", "Surface", eid="s-2"),
                _mk_entity("wing", "Surface", eid="s-1"),
            ]
        },
    }

    out = materialize_entities_in_place(copy.deepcopy(params))
    a_list = out["a"]["stored_entities"]
    b_list = out["b"]["stored_entities"]

    # Objects with same (type, id) across different lists should be the same instance
    assert a_list[0] is b_list[1]
    assert a_list[1] is b_list[0]


def test_per_list_dedup_for_non_point():
    """
    Test: Materializer deduplicates non-Point entities within each list.

    Purpose:
    - Verify that materialize_entities_in_place() removes duplicate entities
    - Verify that deduplication is based on stable key (type, id) tuple
    - Verify that order is preserved (first occurrence kept)
    - Verify that this applies to all non-Point entity types

    Expected behavior:
    - Input: List with duplicate Surface entities (same id "s-1")
    - Process: Deduplication removes second occurrence
    - Output: Single "wing" and one "tail" entity remain

    Note: Point entities are exempt from deduplication (tested separately).
    """
    params = {
        "node": {
            "stored_entities": [
                _mk_entity("wing", "Surface", eid="s-1"),
                _mk_entity("wing", "Surface", eid="s-1"),  # duplicate
                _mk_entity("tail", "Surface", eid="s-2"),
            ]
        }
    }

    out = materialize_entities_in_place(copy.deepcopy(params))
    items = out["node"]["stored_entities"]
    # Dedup preserves order and removes duplicates for non-Point types
    assert [e.name for e in items] == ["wing", "tail"]


def test_skip_dedup_for_point():
    """
    Test: Point entities are exempt from deduplication during materialization.

    Purpose:
    - Verify that Point entity type is explicitly excluded from deduplication
    - Verify that duplicate Point entities with identical data are preserved
    - Verify that this exception only applies to Point (not PointArray, etc.)

    Expected behavior:
    - Input: Two Point entities with same name and location
    - Process: Materialization skips deduplication for Point type
    - Output: Both Point entities remain in the list

    Rationale: Point entities may intentionally be duplicated for different
    use cases (e.g., multiple probes or streamline seeds at same location).
    """
    params = {
        "node": {
            "stored_entities": [
                {
                    "name": "p1",
                    "private_attribute_entity_type_name": "Point",
                    "location": {"units": "m", "value": [0.0, 0.0, 0.0]},
                    "private_attribute_id": "p1ahgdszhf",
                },
                {
                    "name": "p1",
                    "private_attribute_entity_type_name": "Point",
                    "location": {"units": "m", "value": [0.0, 0.0, 0.0]},
                    "private_attribute_id": "p2aaaaaa",
                },  # duplicate Point remains
                {
                    "name": "p2",
                    "private_attribute_entity_type_name": "Point",
                    "location": {"units": "m", "value": [1.0, 0.0, 0.0]},
                    "private_attribute_id": "p3dszahg",
                },
            ]
        }
    }

    out = materialize_entities_in_place(copy.deepcopy(params))
    items = out["node"]["stored_entities"]
    assert [e.name for e in items] == ["p1", "p1", "p2"]


def test_reentrant_safe_and_idempotent():
    """
    Test: Materializer is reentrant-safe and idempotent.

    Purpose:
    - Verify that materialize_entities_in_place() can be called multiple times safely
    - Verify that subsequent calls on already-materialized data are no-ops
    - Verify that object identity is maintained across re-entrant calls
    - Verify that deduplication results are stable

    Expected behavior:
    - First call: Converts dicts to objects, deduplicates
    - Second call: Recognizes already-materialized objects, preserves identity
    - Output: Same results, same object identities (items1[0] is items2[0])

    This property is important for pipeline robustness and allows the
    materializer to be called at multiple stages without side effects.
    """
    params = {
        "node": {
            "stored_entities": [
                _mk_entity("wing", "Surface", eid="s-1"),
                _mk_entity("wing", "Surface", eid="s-1"),  # duplicate
                _mk_entity("tail", "Surface", eid="s-2"),
            ]
        }
    }

    out1 = materialize_entities_in_place(copy.deepcopy(params))
    # Re-entrant call on already materialized objects
    out2 = materialize_entities_in_place(out1)
    items1 = out1["node"]["stored_entities"]
    items2 = out2["node"]["stored_entities"]

    assert [e.name for e in items1] == ["wing", "tail"]
    assert [e.name for e in items2] == ["wing", "tail"]
    # Identity maintained across re-entrant call
    assert items1[0] is items2[0]
    assert items1[1] is items2[1]


def test_materialize_dedup_and_point_passthrough():
    params = {
        "models": [
            {
                "entities": {
                    "stored_entities": [
                        _mk_surface_dict("wing", "s1"),
                        _mk_surface_dict("wing", "s1"),  # duplicate by id
                        _mk_point_dict("p1", "p1", (0.0, 0.0, 0.0)),
                    ]
                }
            }
        ]
    }

    out = materialize_entities_in_place(params)
    items = out["models"][0]["entities"]["stored_entities"]

    # 1) Surfaces are deduped per list
    assert sum(isinstance(x, Surface) for x in items) == 1
    # 2) Points are not deduped by policy (pass-through in not_merged_types)
    assert sum(isinstance(x, Point) for x in items) == 1

    # 3) Idempotency: re-run should keep the same shape and types
    out2 = materialize_entities_in_place(out)
    items2 = out2["models"][0]["entities"]["stored_entities"]
    assert len(items2) == len(items)
    assert sum(isinstance(x, Surface) for x in items2) == 1
    assert sum(isinstance(x, Point) for x in items2) == 1


def test_materialize_passthrough_on_reentrant_call():
    # Re-entrant call should pass through object instances and remain stable
    explicit = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "s1",
            "name": "wing",
        }
    )
    params = {
        "models": [
            {
                "entities": {
                    "stored_entities": [
                        explicit,
                    ]
                }
            }
        ]
    }
    out = materialize_entities_in_place(params)
    items = out["models"][0]["entities"]["stored_entities"]
    assert len([x for x in items if isinstance(x, Surface)]) == 1


def test_materialize_reuses_cached_instance_across_nodes():
    # Same entity appears in multiple lists -> expect the same Python object reused
    sdict = _mk_surface_dict("wing", "s1")
    params = {
        "models": [
            {"entities": {"stored_entities": [sdict]}},
            {"entities": {"stored_entities": [sdict]}},
        ]
    }

    out = materialize_entities_in_place(params)
    items1 = out["models"][0]["entities"]["stored_entities"]
    items2 = out["models"][1]["entities"]["stored_entities"]

    obj1 = next(x for x in items1 if isinstance(x, Surface))
    obj2 = next(x for x in items2 if isinstance(x, Surface))
    # identity check: cached instance is reused across nodes
    assert obj1 is obj2


def test_materialize_with_entity_registry_mode():
    """
    Test: Mode 2 - materialize_entities_in_place with EntityRegistry.

    Purpose:
    - Verify that when EntityRegistry is provided, entities are looked up from registry
    - Verify that params references point to the same instances as in the registry
    - Verify that this enables reference identity between entity_info and params

    Expected behavior:
    - Input: Entity dicts with IDs + pre-populated EntityRegistry
    - Process: Lookup entities by (type, id) from registry
    - Output: Params contain references to registry instances (identity check)
    """
    # Create registry with canonical entity instances
    registry = EntityRegistry()
    wing = Surface(name="wing", private_attribute_id="s-1")
    tail = Surface(name="tail", private_attribute_id="s-2")
    registry.register(wing)
    registry.register(tail)

    # Params with entity dicts referencing the registry entities
    params = {
        "a": {"stored_entities": [_mk_surface_dict("wing", "s-1")]},
        "b": {
            "stored_entities": [
                _mk_surface_dict("tail", "s-2"),
                _mk_surface_dict("wing", "s-1"),
            ]
        },
    }

    # Materialize with registry - should use registry instances
    out = materialize_entities_in_place(copy.deepcopy(params), entity_registry=registry)

    # Verify that params now reference the exact registry instances
    a_wing = out["a"]["stored_entities"][0]
    b_tail = out["b"]["stored_entities"][0]
    b_wing = out["b"]["stored_entities"][1]

    assert a_wing is wing  # Same object from registry
    assert b_tail is tail  # Same object from registry
    assert b_wing is wing  # Same object from registry
    assert a_wing is b_wing  # Same object across different lists


def test_materialize_with_entity_registry_missing_entity_raises():
    """
    Test: Mode 2 - Error when entity not found in registry.

    Purpose:
    - Verify that materialize_entities_in_place raises clear error
    - Verify that error includes entity type, ID, and name for debugging

    Expected behavior:
    - Input: Entity dict with ID not in registry
    - Process: Lookup fails in registry
    - Output: Raise ValueError with descriptive message
    """
    registry = EntityRegistry()
    wing = pd.TypeAdapter(Surface).validate_python({"name": "wing", "private_attribute_id": "s-1"})
    registry.register(wing)

    # Reference to non-existent entity
    params = {"a": {"stored_entities": [_mk_surface_dict("fuselage", "s-999")]}}

    with pytest.raises(ValueError, match=r"Entity not found in EntityRegistry.*s-999.*fuselage"):
        materialize_entities_in_place(copy.deepcopy(params), entity_registry=registry)


def test_materialize_with_entity_registry_missing_id_raises():
    """
    Test: Mode 2 - Error when entity dict missing private_attribute_id.

    Purpose:
    - Verify that all entities must have IDs in registry mode
    - Verify that error message is clear about missing ID requirement

    Expected behavior:
    - Input: Entity dict without private_attribute_id + registry provided
    - Process: Validation detects missing ID
    - Output: Raise ValueError indicating ID is required in registry mode
    """
    registry = EntityRegistry()

    # Entity dict without ID
    params = {
        "a": {
            "stored_entities": [{"name": "wing", "private_attribute_entity_type_name": "Surface"}]
        }
    }

    with pytest.raises(ValueError, match=r"Entity missing 'private_attribute_id'.*EntityRegistry"):
        materialize_entities_in_place(copy.deepcopy(params), entity_registry=registry)
