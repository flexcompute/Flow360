import pydantic as pd

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_in_place,
)
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.primitives import Surface


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
