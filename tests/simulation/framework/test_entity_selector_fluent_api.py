import json

import pytest

from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import (
    expand_entity_list_selectors_in_place,
)
from flow360.component.simulation.primitives import Edge, Surface


class _EntityListStub:
    """Minimal stub for selector expansion tests (avoids EntityList metaclass constraints)."""

    def __init__(self, *, stored_entities=None, selectors=None):
        self.stored_entities = stored_entities or []
        self.selectors = selectors


def _mk_pool(names, entity_type):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": entity_type} for n in names]


def _make_registry(surfaces=None, edges=None):
    """Create an EntityRegistry from entity dictionaries."""
    registry = EntityRegistry()
    for entity_dict in surfaces or []:
        registry.register(Surface(name=entity_dict["name"]))
    for entity_dict in edges or []:
        registry.register(Edge(name=entity_dict["name"]))
    return registry


def _expand_and_get_names(registry: EntityRegistry, selector_model) -> list[str]:
    entity_list = _EntityListStub(stored_entities=[], selectors=[selector_model])
    expand_entity_list_selectors_in_place(registry, entity_list, merge_mode="merge")
    stored = entity_list.stored_entities
    return [
        e.name
        for e in stored
        if e.private_attribute_entity_type_name == selector_model.target_class
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
    registry = _make_registry(
        surfaces=_mk_pool(["wing", "wing-root", "wingtip", "tail"], "Surface")
    )

    # AND logic by default; expect intersection of predicates
    selector = Surface.match("wing*", name="t_and").not_any_of(["wing"])
    names = _expand_and_get_names(registry, selector)
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
    registry = _make_registry(surfaces=_mk_pool(["s1", "s2", "tail", "wing"], "Surface"))

    # OR logic: union of predicates
    selector = Surface.match("s1", name="t_or", logic="OR").any_of(["tail"])
    names = _expand_and_get_names(registry, selector)
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
    registry = _make_registry(surfaces=_mk_pool(["wing", "wing-root", "tail"], "Surface"))

    # Regex fullmatch for exact 'wing', then exclude via not_match (glob)
    selector = Surface.match(r"^wing$", name="t_regex", syntax="regex").not_match(
        "*-root", syntax="glob"
    )
    names = _expand_and_get_names(registry, selector)
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
    registry = _make_registry(surfaces=_mk_pool(["a", "b", "c", "d"], "Surface"))

    # AND semantics: in {a,b,c} and not_in {b}
    selector = Surface.match("*", name="t_in").any_of(["a", "b", "c"]).not_any_of(["b"])
    names = _expand_and_get_names(registry, selector)
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
    registry = _make_registry(edges=_mk_pool(["edgeA", "edgeB"], "Edge"))

    selector = Edge.match("edgeA", name="edge_basic")
    entity_list = _EntityListStub(stored_entities=[], selectors=[selector])
    expand_entity_list_selectors_in_place(registry, entity_list, merge_mode="merge")
    stored = entity_list.stored_entities
    assert [e.name for e in stored if e.private_attribute_entity_type_name == "Edge"] == ["edgeA"]


def test_selector_factory_propagates_description():
    """
    Test: SelectorFactory methods propagate description into EntitySelector instances.

    Expected behavior:
    - Passing description to Surface.match() stores it on the resulting selector.
    - model_dump() contains the provided description for serialization/round-trip.
    """
    selector = Surface.match("*", name="desc_selector", description="Select all surfaces")
    assert selector.description == "Select all surfaces"
    assert selector.model_dump()["description"] == "Select all surfaces"
