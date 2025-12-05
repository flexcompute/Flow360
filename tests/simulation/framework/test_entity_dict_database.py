"""
Tests for entity selector and get_selector_pool_from_dict function.
"""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from flow360.component.simulation.framework.entity_expansion_utils import (
    build_entity_pool_from_entity_info,
    get_selector_pool_from_dict,
    get_selector_pool_from_params,
)
from flow360.component.simulation.framework.entity_materializer import (
    _stable_entity_key_from_obj,
)
from flow360.component.simulation.framework.entity_selector import SelectorEntityPool
from flow360.component.simulation.primitives import Box, GenericVolume, Surface


def _load_simulation_json(relative_path: str) -> dict:
    """Helper function to load simulation JSON files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(test_dir, "..", relative_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class _DummyParams:
    """Minimal SimulationParams-like object for database helpers."""

    def __init__(self, params_dict: dict, entity_info_obj=None):
        self._params_dict = copy.deepcopy(params_dict)
        asset_cache_dict = self._params_dict.get("private_attribute_asset_cache", {})
        if entity_info_obj is None:
            entity_info_dict = asset_cache_dict.get("project_entity_info", {})
            entity_info_obj = SimpleNamespace(
                type_name=entity_info_dict.get("type_name"),
                boundaries=[
                    SimpleNamespace(**boundary)
                    for boundary in entity_info_dict.get("boundaries", [])
                ],
                zones=[SimpleNamespace(**zone) for zone in entity_info_dict.get("zones", [])],
                face_attribute_names=entity_info_dict.get("face_attribute_names"),
                grouped_faces=entity_info_dict.get("grouped_faces"),
                face_group_tag=entity_info_dict.get("face_group_tag"),
                edge_attribute_names=entity_info_dict.get("edge_attribute_names"),
                grouped_edges=entity_info_dict.get("grouped_edges"),
                edge_group_tag=entity_info_dict.get("edge_group_tag"),
                body_attribute_names=entity_info_dict.get("body_attribute_names"),
                grouped_bodies=entity_info_dict.get("grouped_bodies"),
                body_group_tag=entity_info_dict.get("body_group_tag"),
            )
        selectors = asset_cache_dict.get("selectors")
        self.private_attribute_asset_cache = SimpleNamespace(
            project_entity_info=entity_info_obj,
            selectors=selectors,
        )

    def model_dump(self, **kwargs):
        return copy.deepcopy(self._params_dict)


def _entity_names(entries):
    return [
        entry["name"] if isinstance(entry, dict) else getattr(entry, "name", None)
        for entry in entries
    ]


def _build_simple_params_dict():
    return {
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "VolumeMeshEntityInfo",
                "boundaries": [
                    {"name": "wall", "private_attribute_entity_type_name": "Surface"},
                    {"name": "sym", "private_attribute_entity_type_name": "Surface"},
                ],
                "zones": [
                    {"name": "zone-1", "private_attribute_entity_type_name": "GenericVolume"}
                ],
            }
        }
    }


def test_get_selector_pool_for_geometry_entity_info():
    """
    Test get_selector_pool_from_dict with GeometryEntityInfo.
    Uses geometry_grouped_by_file/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_selector_pool_from_dict(params_as_dict)

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

    assert isinstance(entity_db, SelectorEntityPool)
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


def test_get_selector_pool_for_volume_mesh_entity_info():
    """
    Test get_selector_pool_from_dict with VolumeMeshEntityInfo.
    Uses vm_entity_provider/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/vm_entity_provider/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_selector_pool_from_dict(params_as_dict)

    # Get expected counts from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))
    expected_zones_count = len(entity_info.get("zones", []))

    assert isinstance(entity_db, SelectorEntityPool)
    assert len(entity_db.surfaces) == expected_boundaries_count
    assert len(entity_db.generic_volumes) == expected_zones_count
    assert len(entity_db.edges) == 0
    assert len(entity_db.geometry_body_groups) == 0

    # Verify entity type names if entities exist
    if entity_db.surfaces:
        assert entity_db.surfaces[0]["private_attribute_entity_type_name"] == "Surface"
    if entity_db.generic_volumes:
        assert entity_db.generic_volumes[0]["private_attribute_entity_type_name"] == "GenericVolume"


def test_get_selector_pool_for_surface_mesh_entity_info():
    """
    Test get_selector_pool_from_dict with SurfaceMeshEntityInfo.
    Uses params/data/surface_mesh/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("params/data/surface_mesh/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_selector_pool_from_dict(params_as_dict)

    # Get expected count from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))

    assert isinstance(entity_db, SelectorEntityPool)
    assert len(entity_db.surfaces) == expected_boundaries_count
    assert len(entity_db.edges) == 0
    assert len(entity_db.geometry_body_groups) == 0
    assert len(entity_db.generic_volumes) == 0

    # Verify entity type name if entities exist
    if entity_db.surfaces:
        assert entity_db.surfaces[0]["private_attribute_entity_type_name"] == "Surface"


def test_get_selector_pool_missing_asset_cache():
    """
    Test that the function raises ValueError when private_attribute_asset_cache is missing.
    """
    params_as_dict = {}

    with pytest.raises(ValueError, match="private_attribute_asset_cache not found"):
        get_selector_pool_from_dict(params_as_dict)


def test_get_selector_pool_missing_entity_info():
    """
    Test that the function raises ValueError when project_entity_info is missing.
    """
    params_as_dict = {"private_attribute_asset_cache": {}}

    with pytest.raises(ValueError, match="project_entity_info not found"):
        get_selector_pool_from_dict(params_as_dict)


def test_geometry_entity_info_respects_grouping_tags():
    """
    Test that GeometryEntityInfo uses the correct grouping tags to extract entities.
    Verifies the function extracts entities based on the set grouping tag.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    entity_db = get_selector_pool_from_dict(params_as_dict)

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


def test_get_selector_pool_from_params_instances_matches_dict():
    params_as_dict = _build_simple_params_dict()
    dummy_params = _DummyParams(params_as_dict)

    dict_db = get_selector_pool_from_dict(params_as_dict)
    instance_db = get_selector_pool_from_params(dummy_params, use_instances=True)

    assert isinstance(instance_db, SelectorEntityPool)
    assert _entity_names(dict_db.surfaces) == _entity_names(instance_db.surfaces)
    assert _entity_names(dict_db.edges) == _entity_names(instance_db.edges)
    assert _entity_names(dict_db.geometry_body_groups) == _entity_names(
        instance_db.geometry_body_groups
    )
    assert _entity_names(dict_db.generic_volumes) == _entity_names(instance_db.generic_volumes)


def test_get_selector_pool_from_params_default_matches_dict():
    params_as_dict = _build_simple_params_dict()
    dummy_params = _DummyParams(params_as_dict)

    dict_db = get_selector_pool_from_dict(params_as_dict)
    default_db = get_selector_pool_from_params(dummy_params)

    assert isinstance(default_db, SelectorEntityPool)
    assert default_db.surfaces == dict_db.surfaces
    assert default_db.edges == dict_db.edges
    assert default_db.generic_volumes == dict_db.generic_volumes
    assert default_db.geometry_body_groups == dict_db.geometry_body_groups


# ============================================================================
# Tests for build_entity_pool_from_entity_info
# ============================================================================


def _create_mock_volume_mesh_entity_info():
    """Create a mock VolumeMeshEntityInfo with actual entity instances."""
    import pydantic as pd

    surface1 = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "surf-001",
            "name": "wall",
        }
    )
    surface2 = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "surf-002",
            "name": "inlet",
        }
    )
    zone1 = pd.TypeAdapter(GenericVolume).validate_python(
        {
            "private_attribute_entity_type_name": "GenericVolume",
            "private_attribute_id": "zone-001",
            "name": "fluid",
        }
    )
    box1 = pd.TypeAdapter(Box).validate_python(
        {
            "private_attribute_entity_type_name": "Box",
            "private_attribute_id": "box-001",
            "name": "refinement",
            "center": {"value": [0.0, 0.0, 0.0], "units": "m"},
            "size": {"value": [1.0, 1.0, 1.0], "units": "m"},
            "axis_of_rotation": [0.0, 0.0, 1.0],
            "angle_of_rotation": {"value": 0.0, "units": "degree"},
        }
    )

    return SimpleNamespace(
        type_name="VolumeMeshEntityInfo",
        boundaries=[surface1, surface2],
        zones=[zone1],
        draft_entities=[box1],
        ghost_entities=[],
    )


def _create_mock_geometry_entity_info():
    """Create a mock GeometryEntityInfo with actual entity instances."""
    import pydantic as pd

    from flow360.component.simulation.primitives import Edge, GeometryBodyGroup

    surface1 = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "face-001",
            "name": "wing_upper",
        }
    )
    surface2 = pd.TypeAdapter(Surface).validate_python(
        {
            "private_attribute_entity_type_name": "Surface",
            "private_attribute_id": "face-002",
            "name": "wing_lower",
        }
    )
    edge1 = pd.TypeAdapter(Edge).validate_python(
        {
            "private_attribute_entity_type_name": "Edge",
            "private_attribute_id": "edge-001",
            "name": "leading_edge",
        }
    )
    body1 = pd.TypeAdapter(GeometryBodyGroup).validate_python(
        {
            "private_attribute_entity_type_name": "GeometryBodyGroup",
            "private_attribute_id": "body-001",
            "name": "wing",
            "private_attribute_tag_key": "bodyId",
            "private_attribute_sub_components": ["comp-1", "comp-2"],
        }
    )

    return SimpleNamespace(
        type_name="GeometryEntityInfo",
        grouped_faces=[[surface1, surface2]],  # One grouping level
        grouped_edges=[[edge1]],
        grouped_bodies=[[body1]],
        face_attribute_names=["faceId"],
        edge_attribute_names=["edgeId"],
        body_attribute_names=["bodyId"],
        face_group_tag="faceId",
        edge_group_tag="edgeId",
        body_group_tag="bodyId",
        edge_ids=[1],  # Non-empty to enable edge extraction
        draft_entities=[],
        ghost_entities=[],
    )


def test_build_entity_pool_from_volume_mesh_entity_info():
    """
    Test build_entity_pool_from_entity_info with VolumeMeshEntityInfo.

    Verifies:
    - Pool contains all boundaries (surfaces)
    - Pool contains all zones (generic volumes)
    - Pool contains draft entities
    - Keys are (type_name, private_attribute_id) tuples
    - Values are the exact same entity instances
    """
    entity_info = _create_mock_volume_mesh_entity_info()
    pool = build_entity_pool_from_entity_info(entity_info)

    # Should have 2 surfaces + 1 zone + 1 box = 4 entities
    assert len(pool) == 4

    # Verify surfaces are in pool with correct keys
    for surface in entity_info.boundaries:
        key = _stable_entity_key_from_obj(surface)
        assert key in pool
        assert pool[key] is surface  # Same instance

    # Verify zone is in pool
    for zone in entity_info.zones:
        key = _stable_entity_key_from_obj(zone)
        assert key in pool
        assert pool[key] is zone

    # Verify draft entities are in pool
    for draft_entity in entity_info.draft_entities:
        key = _stable_entity_key_from_obj(draft_entity)
        assert key in pool
        assert pool[key] is draft_entity


def test_build_entity_pool_from_geometry_entity_info():
    """
    Test build_entity_pool_from_entity_info with GeometryEntityInfo.

    Verifies:
    - Pool contains all grouped faces (surfaces)
    - Pool contains all grouped edges
    - Pool contains all grouped bodies
    - Handles nested list structure (grouped_faces is list of lists)
    """
    entity_info = _create_mock_geometry_entity_info()
    pool = build_entity_pool_from_entity_info(entity_info)

    # Should have 2 surfaces + 1 edge + 1 body = 4 entities
    assert len(pool) == 4

    # Verify surfaces from grouped_faces
    for group in entity_info.grouped_faces:
        for surface in group:
            key = _stable_entity_key_from_obj(surface)
            assert key in pool
            assert pool[key] is surface

    # Verify edges from grouped_edges
    for group in entity_info.grouped_edges:
        for edge in group:
            key = _stable_entity_key_from_obj(edge)
            assert key in pool
            assert pool[key] is edge

    # Verify bodies from grouped_bodies
    for group in entity_info.grouped_bodies:
        for body in group:
            key = _stable_entity_key_from_obj(body)
            assert key in pool
            assert pool[key] is body


def test_build_entity_pool_includes_ghost_entities():
    """
    Test that build_entity_pool_from_entity_info includes ghost entities.
    """
    import pydantic as pd

    from flow360.component.simulation.primitives import GhostSphere

    ghost = pd.TypeAdapter(GhostSphere).validate_python(
        {
            "private_attribute_entity_type_name": "GhostSphere",
            "private_attribute_id": "ghost-001",
            "name": "farfield",
            "center": [0.0, 0.0, 0.0],
            "maxRadius": 100.0,
        }
    )

    entity_info = SimpleNamespace(
        type_name="VolumeMeshEntityInfo",
        boundaries=[],
        zones=[],
        draft_entities=[],
        ghost_entities=[ghost],
    )

    pool = build_entity_pool_from_entity_info(entity_info)

    assert len(pool) == 1
    key = _stable_entity_key_from_obj(ghost)
    assert key in pool
    assert pool[key] is ghost


def test_build_entity_pool_empty_entity_info():
    """
    Test build_entity_pool_from_entity_info with empty entity_info.
    """
    entity_info = SimpleNamespace(
        type_name="VolumeMeshEntityInfo",
        boundaries=[],
        zones=[],
        draft_entities=[],
        ghost_entities=[],
    )

    pool = build_entity_pool_from_entity_info(entity_info)
    assert len(pool) == 0
    assert pool == {}
