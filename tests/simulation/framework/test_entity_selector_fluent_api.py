import json

import pytest

from flow360.component.simulation.framework.entity_selector import (
    EntityDictDatabase,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.primitives import Edge, Surface


def _mk_pool(names, entity_type):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": entity_type} for n in names]


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
    """
    Test: EntitySelector fluent API with AND logic (default) and predicate chaining.

    Purpose:
    - Verify that Surface.match() creates a selector with glob pattern matching
    - Verify that chaining .not_any_of() adds an exclusion predicate
    - Verify that AND logic correctly computes intersection of predicates
    - Verify that the selector expands correctly against an entity database

    Expected behavior:
    - match("wing*") selects: ["wing", "wing-root", "wingtip"]
    - not_any_of(["wing"]) excludes: ["wing"]
    - AND logic result: ["wing-root", "wingtip"]
    """
    # Prepare a pool of Surface entities
    db = EntityDictDatabase(surfaces=_mk_pool(["wing", "wing-root", "wingtip", "tail"], "Surface"))

    # AND logic by default; expect intersection of predicates
    selector = Surface.match("wing*", name="t_and").not_any_of(["wing"])
    names = _expand_and_get_names(db, selector)
    assert names == ["wing-root", "wingtip"]


def test_surface_class_match_or_union():
    """
    Test: EntitySelector with OR logic for union of predicates.

    Purpose:
    - Verify that logic="OR" parameter works correctly
    - Verify that OR logic computes union of all matching predicates
    - Verify that result order is stable (preserved from original pool)
    - Verify that any_of() predicate works in OR mode

    Expected behavior:
    - match("s1") selects: ["s1"]
    - any_of(["tail"]) selects: ["tail"]
    - OR logic result: ["s1", "tail"] (in pool order)
    """
    db = EntityDictDatabase(surfaces=_mk_pool(["s1", "s2", "tail", "wing"], "Surface"))

    # OR logic: union of predicates
    selector = Surface.match("s1", name="t_or", logic="OR").any_of(["tail"])
    names = _expand_and_get_names(db, selector)
    # Order preserved by pool scan under OR
    assert names == ["s1", "tail"]


def test_surface_regex_and_not_match():
    """
    Test: EntitySelector with mixed syntax (regex and glob) for pattern matching.

    Purpose:
    - Verify that syntax="regex" enables regex pattern matching (fullmatch)
    - Verify that syntax="glob" enables glob pattern matching (default)
    - Verify that match() and not_match() predicates can be chained
    - Verify that different syntax modes can be used in the same selector

    Expected behavior:
    - match(r"^wing$", syntax="regex") selects: ["wing"] (exact match)
    - not_match("*-root", syntax="glob") excludes: ["wing-root"]
    - Result: ["wing"] (passed both predicates)
    """
    db = EntityDictDatabase(surfaces=_mk_pool(["wing", "wing-root", "tail"], "Surface"))

    # Regex fullmatch for exact 'wing', then exclude via not_match (glob)
    selector = Surface.match(r"^wing$", name="t_regex", syntax="regex").not_match(
        "*-root", syntax="glob"
    )
    names = _expand_and_get_names(db, selector)
    assert names == ["wing"]


def test_in_and_not_any_of_chain():
    """
    Test: EntitySelector with any_of() and not_any_of() membership predicates.

    Purpose:
    - Verify that any_of() (inclusion) predicate works correctly
    - Verify that not_any_of() (exclusion) predicate works correctly
    - Verify that membership predicates can be combined with pattern matching
    - Verify that AND logic correctly applies set operations in sequence

    Expected behavior:
    - match("*") selects all: ["a", "b", "c", "d"]
    - any_of(["a", "b", "c"]) filters to: ["a", "b", "c"]
    - not_any_of(["b"]) excludes: ["b"]
    - Final result: ["a", "c"]
    """
    db = EntityDictDatabase(surfaces=_mk_pool(["a", "b", "c", "d"], "Surface"))

    # AND semantics: in {a,b,c} and not_in {b}
    selector = Surface.match("*", name="t_in").any_of(["a", "b", "c"]).not_any_of(["b"])
    names = _expand_and_get_names(db, selector)
    assert names == ["a", "c"]


def test_edge_class_basic_match():
    """
    Test: EntitySelector with Edge entity type (non-Surface).

    Purpose:
    - Verify that entity selector works with different entity types (Edge vs Surface)
    - Verify that Edge.match() creates a selector targeting Edge entities
    - Verify that the entity database correctly routes to the edges pool
    - Verify that simple exact name matching works

    Expected behavior:
    - Edge.match("edgeA") selects only edgeA from the edges pool
    - Edge entities are correctly filtered by target_class
    """
    db = EntityDictDatabase(edges=_mk_pool(["edgeA", "edgeB"], "Edge"))

    selector = Edge.match("edgeA", name="edge_basic")
    params = {"node": {"selectors": [selector.model_dump()]}}
    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]
    assert [e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Edge"] == [
        "edgeA"
    ]
