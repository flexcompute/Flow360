from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import (
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.primitives import Edge, Surface


def _mk_pool(names, entity_type):
    return [{"name": n, "private_attribute_entity_type_name": entity_type} for n in names]


def _make_registry(surfaces=None, edges=None):
    """Create an EntityRegistry from entity dictionaries."""
    registry = EntityRegistry()
    for entity_dict in surfaces or []:
        registry.register(Surface(name=entity_dict["name"]))
    for entity_dict in edges or []:
        registry.register(Edge(name=entity_dict["name"]))
    return registry


def test_merge_mode_preserves_explicit_then_appends_selector_results():
    registry = _make_registry(surfaces=_mk_pool(["wing", "tail", "body"], "Surface"))
    params = {
        "node": {
            "stored_entities": [Surface(name="tail")],
            "selectors": [
                {
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "any_of", "value": ["wing"]}],
                }
            ],
        }
    }
    expand_entity_selectors_in_place(registry, params, merge_mode="merge")
    items = params["node"]["stored_entities"]
    assert [e.name for e in items if e.private_attribute_entity_type_name == "Surface"] == [
        "tail",
        "wing",
    ]
    assert params["node"]["selectors"] == [
        {
            "target_class": "Surface",
            "children": [{"attribute": "name", "operator": "any_of", "value": ["wing"]}],
        }
    ]


def test_replace_mode_overrides_target_class_only():
    registry = _make_registry(
        surfaces=_mk_pool(["wing", "tail"], "Surface"),
        edges=_mk_pool(["e1"], "Edge"),
    )
    params = {
        "node": {
            "stored_entities": [
                Surface(name="tail"),
                Edge(name="e1"),
            ],
            "selectors": [
                {
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "any_of", "value": ["wing"]}],
                }
            ],
        }
    }
    expand_entity_selectors_in_place(registry, params, merge_mode="replace")
    items = params["node"]["stored_entities"]
    # Surface entries replaced by selector result; Edge preserved
    assert [e.name for e in items if e.private_attribute_entity_type_name == "Surface"] == ["wing"]
    assert [e.name for e in items if e.private_attribute_entity_type_name == "Edge"] == ["e1"]
    assert params["node"]["selectors"] == [
        {
            "children": [{"attribute": "name", "operator": "any_of", "value": ["wing"]}],
            "target_class": "Surface",
        }
    ]
