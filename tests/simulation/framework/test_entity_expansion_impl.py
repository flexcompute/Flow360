import copy
import json
import os

from flow360.component.simulation.framework.entity_selector import (
    EntityDictDatabase,
    _compile_glob_cached,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.services import resolve_selectors


def _mk_pool(names, entity_type):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": entity_type} for n in names]


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
    db = EntityDictDatabase(surfaces=_mk_pool(pool_names, "Surface"))

    # Build selectors that cover operators和regex/glob（每条注释标注该 selector 的期望匹配，按池顺序）
    params = {
        "node": {
            "selectors": [
                {
                    # any_of(["tail"]) -> ["tail"]
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "any_of", "value": ["tail"]}],
                },
                {
                    # not_any_of(["wing"]) -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "not_any_of", "value": ["wing"]}
                    ],
                },
                {
                    # any_of(["wing","fuselage"]) -> ["wing","fuselage"]
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "any_of", "value": ["wing", "fuselage"]}
                    ],
                },
                {
                    # not_any_of(["tail","hinge"]) -> ["wing","wingtip","wing-root","wind","tailplane","fuselage","body","leading-wing","my_wing"]
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "not_any_of", "value": ["tail", "hinge"]}
                    ],
                },
                {
                    # matches("wing*") -> ["wing","wingtip","wing-root"]
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "matches", "value": "wing*"}],
                },
                {
                    # not_matches("^wing$", regex) -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
                    "target_class": "Surface",
                    "children": [
                        {
                            "attribute": "name",
                            "operator": "not_matches",
                            "value": "^wing$",
                            "non_glob_syntax": "regex",
                        }
                    ],
                },
            ]
        }
    }

    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]

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

    final_names = [
        e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Surface"
    ]
    assert final_names == expected


def test_combined_predicates_and_or():
    db = EntityDictDatabase(surfaces=_mk_pool(["s1", "s2", "wing", "wing-root", "tail"], "Surface"))

    params = {
        "node": {
            "selectors": [
                {
                    "target_class": "Surface",
                    "logic": "AND",
                    "children": [
                        {"attribute": "name", "operator": "matches", "value": "wing*"},
                        {"attribute": "name", "operator": "not_any_of", "value": ["wing"]},
                    ],
                },
                {
                    "target_class": "Surface",
                    "logic": "OR",
                    "children": [
                        {"attribute": "name", "operator": "any_of", "value": ["s1"]},
                        {"attribute": "name", "operator": "any_of", "value": ["tail"]},
                    ],
                },
                {
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "any_of", "value": ["wing", "wing-root"]},
                    ],
                },
            ]
        }
    }

    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]

    # Union across three selectors (concatenated in selector order, no dedup):
    # 1) AND wing* & notIn ["wing"] -> ["wing-root"]
    # 2) OR in ["s1"] or in ["tail"] -> ["s1", "tail"]
    # 3) default AND with in {wing, wing-root} -> ["wing", "wing-root"]
    # Final list -> ["wing-root", "s1", "tail", "wing", "wing-root"]
    final_names = [
        e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Surface"
    ]
    assert final_names == ["wing-root", "s1", "tail", "wing", "wing-root"]


def test_attribute_tag_scalar_support():
    # Entities include an additional scalar attribute 'tag'
    surfaces = [
        {"name": "wing", "tag": "A", "private_attribute_entity_type_name": "Surface"},
        {"name": "tail", "tag": "B", "private_attribute_entity_type_name": "Surface"},
        {"name": "fuselage", "tag": "A", "private_attribute_entity_type_name": "Surface"},
    ]
    db = EntityDictDatabase(surfaces=surfaces)

    # Use attribute 'tag' in predicates (engine should not assume 'name')
    params = {
        "node": {
            "selectors": [
                {
                    "target_class": "Surface",
                    "logic": "AND",
                    "children": [
                        {"attribute": "tag", "operator": "any_of", "value": ["A"]},
                    ],
                },
                {
                    "target_class": "Surface",
                    "logic": "OR",
                    "children": [
                        {"attribute": "tag", "operator": "any_of", "value": ["B"]},
                        {"attribute": "tag", "operator": "matches", "value": "A"},
                    ],
                },
            ]
        }
    }

    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]
    # Expect union of two selectors:
    # 1) AND tag in ["A"] -> [wing, fuselage]
    # 2) OR tag in {B} or matches 'A' -> pool-order union -> [wing, tail, fuselage]
    final_names = [
        e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Surface"
    ]
    assert final_names == ["wing", "fuselage", "wing", "tail", "fuselage"]


def test_service_expand_entity_selectors_in_place_end_to_end():
    # Pick a complex simulation.json as input
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(test_dir, "..", "data", "geometry_grouped_by_file", "simulation.json")
    with open(sim_path, "r", encoding="utf-8") as fp:
        params = json.load(fp)

    # Convert first output's entities to use a wildcard selector and clear stored entities
    outputs = params.get("outputs") or []
    if not outputs:
        return
    entities = outputs[0].get("entities") or {}
    entities["selectors"] = [
        {
            "target_class": "Surface",
            "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
        }
    ]
    entities["stored_entities"] = []
    outputs[0]["entities"] = entities

    # Expand via service function
    expanded = json.loads(json.dumps(params))
    resolve_selectors(expanded)

    # Build or load a reference file (only created if missing)
    ref_dir = os.path.join(test_dir, "..", "ref")
    ref_path = os.path.join(ref_dir, "entity_expansion_service_ref_outputs.json")

    # Load reference and compare with expanded outputs
    with open(ref_path, "r", encoding="utf-8") as fp:
        ref_outputs = json.load(fp)
    assert compare_values(expanded.get("outputs"), ref_outputs)


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
        regex = _compile_glob_cached(pattern)
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
