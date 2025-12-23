"""
Tests for GeometryEntityInfo all_*_ids properties.

These tests ensure that the prioritized path (using `bodies_face_edge_ids`) produces the
same outputs as the fallback path (using `face_ids` / `edge_ids` / `body_ids`).
"""

import copy
import json
import os

from flow360.component.simulation.entity_info import (
    GeometryEntityInfo,
    parse_entity_info_model,
)


def _load_simulation_json(relative_path: str) -> dict:
    """Load a simulation JSON file relative to this test folder."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(test_dir, "..", relative_path)
    with open(json_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def test_geometry_entity_info_all_ids_match_fallback_when_bodies_face_edge_ids_removed():
    """
    `all_face_ids`, `all_edge_ids`, and `all_body_ids` should be consistent between:
    - prioritized mode: `bodies_face_edge_ids` is present
    - fallback mode: `bodies_face_edge_ids` removed so the legacy lists are used
    """
    params_as_dict = _load_simulation_json("data/geometry_airplane/simulation.json")
    entity_info_dict = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]

    entity_info_prioritized = parse_entity_info_model(entity_info_dict)
    assert isinstance(entity_info_prioritized, GeometryEntityInfo)
    assert entity_info_prioritized.bodies_face_edge_ids is not None

    prioritized_faces = entity_info_prioritized.all_face_ids
    prioritized_edges = entity_info_prioritized.all_edge_ids
    prioritized_bodies = entity_info_prioritized.all_body_ids

    entity_info_dict_fallback = copy.deepcopy(entity_info_dict)
    entity_info_dict_fallback.pop("bodies_face_edge_ids", None)

    entity_info_fallback = parse_entity_info_model(entity_info_dict_fallback)
    assert isinstance(entity_info_fallback, GeometryEntityInfo)
    assert entity_info_fallback.bodies_face_edge_ids is None

    assert entity_info_fallback.all_face_ids == prioritized_faces
    assert entity_info_fallback.all_edge_ids == prioritized_edges
    assert entity_info_fallback.all_body_ids == prioritized_bodies
