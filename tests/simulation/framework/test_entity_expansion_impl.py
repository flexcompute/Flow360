import copy
import json
import os

from flow360.component.simulation.framework.entity_expansion_utils import (
    expand_all_entity_lists_in_place,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import (
    compile_glob_cached,
    expand_entity_list_selectors_in_place,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.primitives import Edge, GenericVolume, Surface
from flow360.component.simulation.simulation_params import SimulationParams


class _EntityListStub:
    """Minimal stub for selector expansion tests (avoids EntityList metaclass constraints)."""

    def __init__(self, *, stored_entities=None, selectors=None):
        self.stored_entities = stored_entities or []
        self.selectors = selectors


def _mk_pool(names, entity_type):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": entity_type} for n in names]


def _make_registry(surfaces=None, edges=None, generic_volumes=None, geometry_body_groups=None):
    """Create an EntityRegistry from entity dictionaries."""
    registry = EntityRegistry()
    for entity_dict in surfaces or []:
        registry.register(Surface(name=entity_dict["name"]))
    for entity_dict in edges or []:
        registry.register(Edge(name=entity_dict["name"]))
    for entity_dict in generic_volumes or []:
        registry.register(GenericVolume(name=entity_dict["name"]))
    # GeometryBodyGroup is handled separately if needed
    return registry


def test_operator_and_syntax_coverage():
    # Prepare database with diverse names (not just quantity)
    pool_names = [
        "wing",
        "wingtip",
        "wing-root",
        "wind",
        "tail",
        "tailplane",
        "fuselage",
        "body",
        "leading-wing",
        "my_wing",
        "hinge",
    ]
    registry = _make_registry(surfaces=_mk_pool(pool_names, "Surface"))

    # Build selectors that cover operators和regex/glob
    entity_list = _EntityListStub(
        stored_entities=[],
        selectors=[
            # any_of(["tail"]) -> ["tail"]
            Surface.any_of(["tail"], name="sel_any_tail"),
            # not_any_of(["wing"]) -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
            Surface.not_any_of(["wing"], name="sel_not_any_wing"),
            # any_of(["wing","fuselage"]) -> ["wing","fuselage"]
            Surface.any_of(["wing", "fuselage"], name="sel_any_wing_fuselage"),
            # not_any_of(["tail","hinge"]) -> ["wing","wingtip","wing-root","wind","tailplane","fuselage","body","leading-wing","my_wing"]
            Surface.not_any_of(["tail", "hinge"], name="sel_not_any_tail_hinge"),
            # matches("wing*") -> ["wing","wingtip","wing-root"]
            Surface.match("wing*", name="sel_match_wing_glob"),
            # not_matches("^wing$", regex) -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
            Surface.not_match("^wing$", name="sel_not_match_exact_wing_regex", syntax="regex"),
        ],
    )

    expand_entity_list_selectors_in_place(registry, entity_list, merge_mode="merge")
    stored = entity_list.stored_entities

    # Build expected union by concatenating each selector's expected results (order matters)
    expected = []
    expected += ["tail"]
    expected += [
        "wingtip",
        "wing-root",
        "wind",
        "tail",
        "tailplane",
        "fuselage",
        "body",
        "leading-wing",
        "my_wing",
        "hinge",
    ]
    expected += ["wing", "fuselage"]
    expected += [
        "wing",
        "wingtip",
        "wing-root",
        "wind",
        "tailplane",
        "fuselage",
        "body",
        "leading-wing",
        "my_wing",
    ]
    expected += ["wing", "wingtip", "wing-root"]
    expected += [
        "wingtip",
        "wing-root",
        "wind",
        "tail",
        "tailplane",
        "fuselage",
        "body",
        "leading-wing",
        "my_wing",
        "hinge",
    ]
    # Note: final_names has been deduplicated by merging.
    final_names = [e.name for e in stored if e.private_attribute_entity_type_name == "Surface"]
    assert sorted(final_names) == sorted(list(set(expected)))


def test_combined_predicates_and_or():
    registry = _make_registry(
        surfaces=_mk_pool(["s1", "s2", "wing", "wing-root", "tail"], "Surface")
    )

    entity_list = _EntityListStub(
        stored_entities=[],
        selectors=[
            Surface.match("wing*", name="sel_and_wing_not_wing", logic="AND").not_any_of(["wing"]),
            Surface.any_of(["s1"], name="sel_or_s1_tail", logic="OR").any_of(["tail"]),
            Surface.any_of(["wing", "wing-root"], name="sel_any_wing_or_root"),
        ],
    )

    expand_entity_list_selectors_in_place(registry, entity_list, merge_mode="merge")
    stored = entity_list.stored_entities

    # Union across three selectors (concatenated in selector order, no dedup):
    # 1) AND wing* & notIn ["wing"] -> ["wing-root"]
    # 2) OR in ["s1"] or in ["tail"] -> ["s1", "tail"]
    # 3) default AND with in {wing, wing-root} -> ["wing", "wing-root"]
    # Final list -> ["wing-root", "s1", "tail", "wing", "wing-root"]
    # Note: final_names has been deduplicated by merging.
    final_names = [e.name for e in stored if e.private_attribute_entity_type_name == "Surface"]
    assert sorted(final_names) == sorted(list(set(["wing-root", "s1", "tail", "wing"])))


def test_compile_glob_cached_extended_syntax_support():
    # Comments in English for maintainers
    # Ensure extended glob features supported by wcmatch translation are honored.
    candidates = [
        "a",
        "b",
        "ab",
        "abc",
        "file",
        "file1",
        "file2",
        "file10",
        "file.txt",
        "File.TXT",
        "data_01",
        "data-xyz",
        "[star]",
        "literal*star",
        "foo.bar",
        ".hidden",
        "1",
        "2",
        "3",
    ]

    def match(pattern: str) -> list[str]:
        regex = compile_glob_cached(pattern)
        return [n for n in candidates if regex.fullmatch(n) is not None]

    # Basic glob
    assert match("file*") == ["file", "file1", "file2", "file10", "file.txt"]
    assert match("file[0-9]") == ["file1", "file2"]

    # Brace expansion
    assert match("{a,b}") == ["a", "b"]
    assert match("file{1,2}") == ["file1", "file2"]
    assert match("{1..3}") == ["1", "2", "3"]
    assert match("file{01..10}") == ["file10"]

    # Extglob
    # In extglob, @(file|data) means exactly 'file' or 'data'. To match 'data_*', use data*.
    assert match("@(file|data*)") == ["file", "data_01", "data-xyz"]
    expected_not_file = [n for n in candidates if n != "file"]
    assert match("!(file)") == expected_not_file
    assert match("?(file)") == ["file"]
    assert match("+(file)") == ["file"]
    assert match("*(file)") == ["file"]

    # POSIX character classes
    assert match("[[:digit:]]*") == ["1", "2", "3"]
    assert match("file[[:digit:]]") == ["file1", "file2"]
    assert match("[[:upper:]]*.[[:alpha:]]*") == ["File.TXT"]

    # Escaping and literals
    assert match("literal[*]star") == ["literal*star"]
    assert match(r"literal\*star") == ["literal*star"]
    assert match(r"foo\.bar") == ["foo.bar"]
    assert match("foo[.]bar") == ["foo.bar"]
