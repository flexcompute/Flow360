import copy
import json
import os

from flow360.component.simulation.framework.entity_selector import (
    EntityDictDatabase,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.services import resolve_selectors


def _mk_pool(names, etype):
    # Build list of entity dicts with given names and type
    return [{"name": n, "private_attribute_entity_type_name": etype} for n in names]


def test_in_place_expansion_and_replacement_per_class():
    # Entity database with two classes
    db = EntityDictDatabase(
        surfaces=_mk_pool(["wing", "tail", "fuselage"], "Surface"),
        edges=_mk_pool(["edgeA", "edgeB"], "Edge"),
    )

    # params_as_dict with existing stored_entities including a non-persistent entity
    params = {
        "outputs": [
            {
                "stored_entities": [
                    {"name": "custom-box", "private_attribute_entity_type_name": "Box"},
                    {"name": "old-wing", "private_attribute_entity_type_name": "Surface"},
                    {"name": "old-edgeA", "private_attribute_entity_type_name": "Edge"},
                ],
                "selectors": [
                    {
                        "target_class": "Surface",
                        "logic": "AND",
                        "children": [
                            {"attribute": "name", "operator": "matches", "value": "w*"},
                        ],
                    },
                    {
                        "target_class": "Edge",
                        "logic": "OR",
                        "children": [
                            {"attribute": "name", "operator": "equals", "value": "edgeB"},
                        ],
                    },
                ],
            }
        ]
    }

    out = expand_entity_selectors_in_place(db, params)

    # In-place: the returned object should be the same reference
    assert out is params

    # Non-persistent entity remains; Surface and Edge replaced by new selection
    stored = params["outputs"][0]["stored_entities"]
    names_by_type = {}
    for e in stored:
        names_by_type.setdefault(e["private_attribute_entity_type_name"], []).append(e["name"])

    assert names_by_type["Box"] == ["custom-box"]
    assert names_by_type["Surface"] == ["wing"]  # matches w*
    assert names_by_type["Edge"] == ["edgeB"]  # equals edgeB

    # selectors cleared
    assert params["outputs"][0]["selectors"] == []


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
                    # equals("tail") -> ["tail"]
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "equals", "value": "tail"}],
                },
                {
                    # notEquals("wing") -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "notEquals", "value": "wing"}],
                },
                {
                    # in(["wing","fuselage"]) -> ["wing","fuselage"]
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "in", "value": ["wing", "fuselage"]}
                    ],
                },
                {
                    # notIn(["tail","hinge"]) -> ["wing","wingtip","wing-root","wind","tailplane","fuselage","body","leading-wing","my_wing"]
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "notIn", "value": ["tail", "hinge"]}
                    ],
                },
                {
                    # matches("wing*") -> ["wing","wingtip","wing-root"]
                    "target_class": "Surface",
                    "children": [{"attribute": "name", "operator": "matches", "value": "wing*"}],
                },
                {
                    # notMatches("^wing$", regex) -> ["wingtip","wing-root","wind","tail","tailplane","fuselage","body","leading-wing","my_wing","hinge"]
                    "target_class": "Surface",
                    "children": [
                        {
                            "attribute": "name",
                            "operator": "notMatches",
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
                        {"attribute": "name", "operator": "notEquals", "value": "wing"},
                    ],
                },
                {
                    "target_class": "Surface",
                    "logic": "OR",
                    "children": [
                        {"attribute": "name", "operator": "equals", "value": "s1"},
                        {"attribute": "name", "operator": "equals", "value": "tail"},
                    ],
                },
                {
                    "target_class": "Surface",
                    "children": [
                        {"attribute": "name", "operator": "in", "value": ["wing", "wing-root"]},
                    ],
                },
            ]
        }
    }

    expand_entity_selectors_in_place(db, params)
    stored = params["node"]["stored_entities"]

    # Union across three selectors (concatenated in selector order, no dedup):
    # 1) AND wing* & notEquals wing -> ["wing-root"]
    # 2) OR equals s1 or tail -> ["s1", "tail"]
    # 3) default AND with in {wing, wing-root} -> ["wing", "wing-root"]
    # Final list -> ["wing-root", "s1", "tail", "wing", "wing-root"]
    final_names = [
        e["name"] for e in stored if e["private_attribute_entity_type_name"] == "Surface"
    ]
    assert final_names == ["wing-root", "s1", "tail", "wing", "wing-root"]


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
