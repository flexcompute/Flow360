import copy

import pytest

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_in_place,
)


def _mk_entity(name: str, entity_type: str, eid: str | None = None) -> dict:
    d = {"name": name, "private_attribute_entity_type_name": entity_type}
    if eid is not None:
        d["private_attribute_id"] = eid
    return d


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
                },
                {
                    "name": "p1",
                    "private_attribute_entity_type_name": "Point",
                    "location": {"units": "m", "value": [0.0, 0.0, 0.0]},
                },  # duplicate Point remains
                {
                    "name": "p2",
                    "private_attribute_entity_type_name": "Point",
                    "location": {"units": "m", "value": [1.0, 0.0, 0.0]},
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
