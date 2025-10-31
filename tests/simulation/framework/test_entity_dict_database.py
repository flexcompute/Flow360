"""
Tests for entity selector and get_entity_database_for_selectors function.
"""

import json
import os

import pytest

from flow360.component.simulation.entity_info import get_entity_database_for_selectors
from flow360.component.simulation.framework.entity_selector import EntityDictDatabase


def _load_simulation_json(relative_path: str) -> dict:
    """Helper function to load simulation JSON files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(test_dir, "..", relative_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_get_entity_database_for_geometry_entity_info():
    """
    Test get_entity_database_for_selectors with GeometryEntityInfo.
    Uses geometry_grouped_by_file/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_entity_database_for_selectors(params_as_dict)

    # Get expected counts from entity_info based on grouping tags
    face_group_tag = entity_info.get("face_group_tag")
    if face_group_tag:
        face_idx = entity_info["face_attribute_names"].index(face_group_tag)
        expected_surfaces_count = len(entity_info["grouped_faces"][face_idx])
    else:
        expected_surfaces_count = 0

    edge_group_tag = entity_info.get("edge_group_tag")
    if edge_group_tag and entity_info.get("edge_ids"):
        edge_idx = entity_info["edge_attribute_names"].index(edge_group_tag)
        expected_edges_count = len(entity_info["grouped_edges"][edge_idx])
    else:
        expected_edges_count = 0

    body_group_tag = entity_info.get("body_group_tag")
    if body_group_tag and entity_info.get("body_attribute_names"):
        body_idx = entity_info["body_attribute_names"].index(body_group_tag)
        expected_bodies_count = len(entity_info["grouped_bodies"][body_idx])
    else:
        expected_bodies_count = 0

    assert isinstance(entity_db, EntityDictDatabase)
    assert len(entity_db.surfaces) == expected_surfaces_count
    assert len(entity_db.edges) == expected_edges_count
    assert len(entity_db.geometry_body_groups) == expected_bodies_count
    assert len(entity_db.generic_volumes) == 0

    # Verify entity type names if entities exist
    if entity_db.surfaces:
        assert entity_db.surfaces[0]["private_attribute_entity_type_name"] == "Surface"
    if entity_db.edges:
        assert entity_db.edges[0]["private_attribute_entity_type_name"] == "Edge"
    if entity_db.geometry_body_groups:
        assert (
            entity_db.geometry_body_groups[0]["private_attribute_entity_type_name"]
            == "GeometryBodyGroup"
        )


def test_get_entity_database_for_volume_mesh_entity_info():
    """
    Test get_entity_database_for_selectors with VolumeMeshEntityInfo.
    Uses vm_entity_provider/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/vm_entity_provider/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_entity_database_for_selectors(params_as_dict)

    # Get expected counts from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))
    expected_zones_count = len(entity_info.get("zones", []))

    assert isinstance(entity_db, EntityDictDatabase)
    assert len(entity_db.surfaces) == expected_boundaries_count
    assert len(entity_db.generic_volumes) == expected_zones_count
    assert len(entity_db.edges) == 0
    assert len(entity_db.geometry_body_groups) == 0

    # Verify entity type names if entities exist
    if entity_db.surfaces:
        assert entity_db.surfaces[0]["private_attribute_entity_type_name"] == "Surface"
    if entity_db.generic_volumes:
        assert entity_db.generic_volumes[0]["private_attribute_entity_type_name"] == "GenericVolume"


def test_get_entity_database_for_surface_mesh_entity_info():
    """
    Test get_entity_database_for_selectors with SurfaceMeshEntityInfo.
    Uses params/data/surface_mesh/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("params/data/surface_mesh/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_entity_database_for_selectors(params_as_dict)

    # Get expected count from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))

    assert isinstance(entity_db, EntityDictDatabase)
    assert len(entity_db.surfaces) == expected_boundaries_count
    assert len(entity_db.edges) == 0
    assert len(entity_db.geometry_body_groups) == 0
    assert len(entity_db.generic_volumes) == 0

    # Verify entity type name if entities exist
    if entity_db.surfaces:
        assert entity_db.surfaces[0]["private_attribute_entity_type_name"] == "Surface"


def test_get_entity_database_missing_asset_cache():
    """
    Test that the function raises ValueError when private_attribute_asset_cache is missing.
    """
    params_as_dict = {}

    with pytest.raises(ValueError, match="private_attribute_asset_cache not found"):
        get_entity_database_for_selectors(params_as_dict)


def test_get_entity_database_missing_entity_info():
    """
    Test that the function raises ValueError when project_entity_info is missing.
    """
    params_as_dict = {"private_attribute_asset_cache": {}}

    with pytest.raises(ValueError, match="project_entity_info not found"):
        get_entity_database_for_selectors(params_as_dict)


def test_geometry_entity_info_respects_grouping_tags():
    """
    Test that GeometryEntityInfo uses the correct grouping tags to extract entities.
    Verifies the function extracts entities based on the set grouping tag.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_entity_database_for_selectors(params_as_dict)

    # Verify face grouping
    face_group_tag = entity_info.get("face_group_tag")
    assert face_group_tag is not None, "Test data should have face_group_tag set"

    face_attribute_names = entity_info.get("face_attribute_names", [])
    grouped_faces = entity_info.get("grouped_faces", [])
    index = face_attribute_names.index(face_group_tag)
    expected_faces = grouped_faces[index]

    assert len(entity_db.surfaces) == len(expected_faces)

    # Verify edge grouping
    if entity_info.get("edge_group_tag"):
        edge_group_tag = entity_info["edge_group_tag"]
        edge_attribute_names = entity_info.get("edge_attribute_names", [])
        grouped_edges = entity_info.get("grouped_edges", [])
        index = edge_attribute_names.index(edge_group_tag)
        expected_edges = grouped_edges[index]
        assert len(entity_db.edges) == len(expected_edges)

    # Verify body grouping
    if entity_info.get("body_group_tag"):
        body_group_tag = entity_info["body_group_tag"]
        body_attribute_names = entity_info.get("body_attribute_names", [])
        grouped_bodies = entity_info.get("grouped_bodies", [])
        index = body_attribute_names.index(body_group_tag)
        expected_bodies = grouped_bodies[index]
        assert len(entity_db.geometry_body_groups) == len(expected_bodies)
