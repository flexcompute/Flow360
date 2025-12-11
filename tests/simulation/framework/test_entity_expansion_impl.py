import copy
import json
import os

from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import (
    compile_glob_cached,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.primitives import Edge, GenericVolume, Surface
from flow360.component.simulation.services import resolve_selectors
from flow360.component.simulation.simulation_params import SimulationParams


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

    expand_entity_selectors_in_place(registry, params)
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

    final_names = [e.name for e in stored if e.private_attribute_entity_type_name == "Surface"]
    assert final_names == expected


def test_combined_predicates_and_or():
    registry = _make_registry(
        surfaces=_mk_pool(["s1", "s2", "wing", "wing-root", "tail"], "Surface")
    )

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

    expand_entity_selectors_in_place(registry, params)
    stored = params["node"]["stored_entities"]

    # Union across three selectors (concatenated in selector order, no dedup):
    # 1) AND wing* & notIn ["wing"] -> ["wing-root"]
    # 2) OR in ["s1"] or in ["tail"] -> ["s1", "tail"]
    # 3) default AND with in {wing, wing-root} -> ["wing", "wing-root"]
    # Final list -> ["wing-root", "s1", "tail", "wing", "wing-root"]
    final_names = [e.name for e in stored if e.private_attribute_entity_type_name == "Surface"]
    assert final_names == ["wing-root", "s1", "tail", "wing", "wing-root"]


def test_attribute_tag_scalar_support():
    # Entities include an additional scalar attribute 'tag'
    # Note: EntityRegistry stores entity objects, so we need to handle this differently
    # For this test, we'll create Surface objects with the tag attribute stored as extra data
    registry = EntityRegistry()
    # Create surfaces with tag attribute via custom approach
    s1 = Surface(name="wing")
    s2 = Surface(name="tail")
    s3 = Surface(name="fuselage")
    # Store tag as a custom attribute
    object.__setattr__(s1, "tag", "A")
    object.__setattr__(s2, "tag", "B")
    object.__setattr__(s3, "tag", "A")
    registry.register(s1)
    registry.register(s2)
    registry.register(s3)

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

    expand_entity_selectors_in_place(registry, params)
    stored = params["node"]["stored_entities"]
    # Expect union of two selectors:
    # 1) AND tag in ["A"] -> [wing, fuselage]
    # 2) OR tag in {B} or matches 'A' -> pool-order union -> [wing, tail, fuselage]
    final_names = [e.name for e in stored if e.private_attribute_entity_type_name == "Surface"]
    assert final_names == ["wing", "fuselage", "wing", "tail", "fuselage"]


def test_service_expand_entity_selectors_in_place_end_to_end():
    # Pick a complex simulation.json as input
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(test_dir, "..", "data", "geometry_grouped_by_file", "simulation.json")
    with open(sim_path, "r", encoding="utf-8") as fp:
        params = json.load(fp)

    params, _ = SimulationParams._update_param_dict(params)

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

    # Convert entity objects to dicts for comparison
    for output in expanded.get("outputs", []):
        entities_obj = output.get("entities", {})
        stored = entities_obj.get("stored_entities", [])
        entities_obj["stored_entities"] = [
            e.model_dump(mode="json", exclude_none=True) if hasattr(e, "model_dump") else e
            for e in stored
        ]

    # Build or load a reference file (only created if missing)
    ref_dir = os.path.join(test_dir, "..", "ref")
    ref_path = os.path.join(ref_dir, "entity_expansion_service_ref_outputs.json")

    # Load reference and compare with expanded outputs
    with open(ref_path, "r", encoding="utf-8") as fp:
        ref_outputs = json.load(fp)
    assert compare_values(expanded.get("outputs"), ref_outputs)


def testcompile_glob_cached_extended_syntax_support():
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
