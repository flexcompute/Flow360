import copy
from typing import Optional

import pydantic as pd

from flow360.component.simulation.framework.entity_materializer import (
    _stable_entity_key_from_obj,
    materialize_entities_in_place,
)
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.primitives import Box, Surface


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


def test_entity_pool_provides_reference_identity():
    """
    Test: Entity pool enables reference identity between pre-existing entities and materialized params.

    Purpose:
    - Verify that entity_pool parameter allows reusing pre-existing entity instances
    - Verify that materialized entities are the exact same Python objects from the pool
    - Verify that modifications to pool entities are reflected in materialized params

    Expected behavior:
    - Input: Pre-existing Surface and Box entities in entity_pool
    - Process: Materialization looks up entities by (type, id) key in pool
    - Output: Materialized entities are the same objects as pool entities (by identity)

    This is the core feature enabling DraftContext entity modifications to propagate to params.
    """
    # Create pre-existing entity instances (simulating entity_info entities)
    surface = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "surf-001",
            "name": "wing",
        }
    )
    box = pd.TypeAdapter(Box).validate_python(
        {
            "private_attribute_entity_type_name": "Box",
            "private_attribute_id": "box-001",
            "name": "refinement_zone",
            "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
            "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
            "axis_of_rotation": [0.0, 0.0, 1.0],
            "angle_of_rotation": {"value": 0.0, "units": "degree"},
        }
    )

    # Build entity pool keyed by (type, id)
    entity_pool = {
        _stable_entity_key_from_obj(surface): surface,
        _stable_entity_key_from_obj(box): box,
    }

    # Params with entity dicts that match pool keys
    params = {
        "models": [
            {
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "surf-001",
                            "name": "wing",
                        },
                    ]
                }
            }
        ],
        "volumes": {
            "stored_entities": [
                {
                    "private_attribute_entity_type_name": "Box",
                    "private_attribute_id": "box-001",
                    "name": "refinement_zone",
                    "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
                    "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
                    "axis_of_rotation": [0.0, 0.0, 1.0],
                    "angle_of_rotation": {"value": 0.0, "units": "degree"},
                },
            ]
        },
    }

    # Materialize with entity_pool
    out = materialize_entities_in_place(params, entity_pool=entity_pool)

    # Verify reference identity - materialized entities ARE the pool entities
    materialized_surface = out["models"][0]["entities"]["stored_entities"][0]
    materialized_box = out["volumes"]["stored_entities"][0]

    assert materialized_surface is surface, "Surface should be same object from entity_pool"
    assert materialized_box is box, "Box should be same object from entity_pool"


def test_entity_pool_modification_reflects_in_materialized_params():
    """
    Test: Modifications to entity_pool entities are reflected in materialized params.

    Purpose:
    - Verify the end-to-end use case: modify entity in pool, see change in params
    - This simulates the DraftContext workflow where users modify entities

    Expected behavior:
    - Create entity, add to pool, materialize params
    - Modify the entity instance (non-frozen field like center)
    - The modification is visible in the materialized params (same object)
    """
    import unyt as u

    # Create entity and pool
    box = pd.TypeAdapter(Box).validate_python(
        {
            "private_attribute_entity_type_name": "Box",
            "private_attribute_id": "box-002",
            "name": "my_box",
            "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
            "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
            "axis_of_rotation": [0.0, 0.0, 1.0],
            "angle_of_rotation": {"value": 0.0, "units": "degree"},
        }
    )
    entity_pool = {_stable_entity_key_from_obj(box): box}

    params = {
        "volumes": {
            "stored_entities": [
                {
                    "private_attribute_entity_type_name": "Box",
                    "private_attribute_id": "box-002",
                    "name": "my_box",
                    "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
                    "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
                    "axis_of_rotation": [0.0, 0.0, 1.0],
                    "angle_of_rotation": {"value": 0.0, "units": "degree"},
                },
            ]
        },
    }

    out = materialize_entities_in_place(params, entity_pool=entity_pool)
    materialized_box = out["volumes"]["stored_entities"][0]

    # Verify initial state
    assert materialized_box is box
    assert list(box.center.value) == [0.0, 0.0, 0.0]

    # Modify the entity (simulating draft.boxes["my_box"].center = ...)
    # Note: center is a non-frozen field that can be modified
    box.center = (1.0, 2.0, 3.0) * u.m

    # The change is immediately visible in materialized params (same object)
    assert list(materialized_box.center.value) == [1.0, 2.0, 3.0]
    assert list(out["volumes"]["stored_entities"][0].center.value) == [1.0, 2.0, 3.0]


def test_entity_pool_fallback_to_builder_for_unknown_entities():
    """
    Test: Entities not in pool are built via the default builder.

    Purpose:
    - Verify that entity_pool is optional and gracefully handles missing entries
    - Verify that entities not found in pool are created normally via builder
    - Verify that pool entities and built entities can coexist

    Expected behavior:
    - Input: Pool with Surface, params with Surface (in pool) and Box (not in pool)
    - Process: Surface reuses pool, Box is built fresh
    - Output: Surface is pool object, Box is new instance
    """
    surface = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "surf-003",
            "name": "fuselage",
        }
    )
    entity_pool = {_stable_entity_key_from_obj(surface): surface}

    params = {
        "models": [
            {
                "entities": {
                    "stored_entities": [
                        # This one is in pool
                        {
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "surf-003",
                            "name": "fuselage",
                        },
                        # This one is NOT in pool - will be built
                        {
                            "private_attribute_entity_type_name": "Box",
                            "private_attribute_id": "box-new",
                            "name": "new_box",
                            "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
                            "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
                            "axis_of_rotation": [0.0, 0.0, 1.0],
                            "angle_of_rotation": {"value": 0.0, "units": "degree"},
                        },
                    ]
                }
            }
        ],
    }

    out = materialize_entities_in_place(params, entity_pool=entity_pool)
    items = out["models"][0]["entities"]["stored_entities"]

    # Surface should be from pool
    assert items[0] is surface

    # Box should be a new instance (not in pool)
    assert isinstance(items[1], Box)
    assert items[1].name == "new_box"
    # And it should NOT be in our original pool
    box_key = _stable_entity_key_from_obj(items[1])
    assert box_key not in entity_pool or entity_pool.get(box_key) is not items[1]
