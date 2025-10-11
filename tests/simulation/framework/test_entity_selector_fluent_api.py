import json

import pytest

from flow360.component.simulation.framework.entity_selector import (
    EntityDictDatabase,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.primitives import Edge, Surface


def _mk_pool(names, etype):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": etype} for n in names]


def _expand_and_get_names(db: EntityDictDatabase, selector_model) -> list[str]:
    # Convert model to dict for the expansion engine
    params = {"node": {"selectors": [selector_model.model_dump()]}}
    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]
    return [
        e["name"]
        for e in stored
        if e["private_attribute_entity_type_name"] == selector_model.target_class
    ]


def test_surface_class_match_and_chain_and():
    # Prepare a pool of Surface entities
    db = EntityDictDatabase(surfaces=_mk_pool(["wing", "wing-root", "wingtip", "tail"], "Surface"))

    # AND logic by default; expect intersection of predicates
    selector = Surface.match("wing*").not_any_of(["wing"])
    names = _expand_and_get_names(db, selector)
    assert names == ["wing-root", "wingtip"]


def test_surface_class_match_or_union():
    db = EntityDictDatabase(surfaces=_mk_pool(["s1", "s2", "tail", "wing"], "Surface"))

    # OR logic: union of predicates
    selector = Surface.match("s1", logic="OR").any_of(["tail"])
    names = _expand_and_get_names(db, selector)
    # Order preserved by pool scan under OR
    assert names == ["s1", "tail"]


def test_surface_regex_and_not_match():
    db = EntityDictDatabase(surfaces=_mk_pool(["wing", "wing-root", "tail"], "Surface"))

    # Regex fullmatch for exact 'wing', then exclude via not_match (glob)
    selector = Surface.match(r"^wing$", syntax="regex").not_match("*-root", syntax="glob")
    names = _expand_and_get_names(db, selector)
    assert names == ["wing"]


def test_in_and_not_any_of_chain():
    db = EntityDictDatabase(surfaces=_mk_pool(["a", "b", "c", "d"], "Surface"))

    # AND semantics: in {a,b,c} and not_in {b}
    selector = Surface.match("*").any_of(["a", "b", "c"]).not_any_of(["b"])
    names = _expand_and_get_names(db, selector)
    assert names == ["a", "c"]


def test_edge_class_basic_match():
    db = EntityDictDatabase(edges=_mk_pool(["edgeA", "edgeB"], "Edge"))

    selector = Edge.match("edgeA")
    params = {"node": {"selectors": [selector.model_dump()]}}
    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]
    assert [e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Edge"] == [
        "edgeA"
    ]
