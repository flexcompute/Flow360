"""
Tests for entity selector and get_entity_info_and_registry_from_dict function.
"""

import copy
import json
import os

import pytest

from flow360.component.simulation.framework.entity_expansion_utils import (
    get_entity_info_and_registry_from_dict,
    get_registry_from_params,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import (
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams


def _load_simulation_json(relative_path: str) -> dict:
    """Helper function to load simulation JSON files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(test_dir, "..", relative_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class _AssetCache:
    """Simple object to hold asset cache data."""

    def __init__(self, project_entity_info, selectors):
        self.project_entity_info = project_entity_info
        self.selectors = selectors


class _DummyParams:
    """Minimal SimulationParams-like object for database helpers."""

    def __init__(self, params_dict: dict, entity_info_obj=None):
        self._params_dict = copy.deepcopy(params_dict)
        asset_cache_dict = self._params_dict.get("private_attribute_asset_cache", {})
        if entity_info_obj is None:
            entity_info_dict = asset_cache_dict.get("project_entity_info", {})
            # Deserialize entity_info_dict to actual entity_info object
            from flow360.component.simulation.entity_info import parse_entity_info_model

            entity_info_obj = parse_entity_info_model(entity_info_dict)

        selectors = asset_cache_dict.get("selectors")
        self.private_attribute_asset_cache = _AssetCache(
            project_entity_info=entity_info_obj, selectors=selectors
        )

    def model_dump(self, **kwargs):
        return copy.deepcopy(self._params_dict)


def _entity_names(entries):
    return [entry["name"] if isinstance(entry, dict) else entry.name for entry in entries]


def _build_simple_params_dict():
    return {
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "VolumeMeshEntityInfo",
                "boundaries": [
                    {
                        "name": "wall",
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_is_interface": False,
                    },
                    {
                        "name": "sym",
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_is_interface": False,
                    },
                ],
                "zones": [
                    {"name": "zone-1", "private_attribute_entity_type_name": "GenericVolume"}
                ],
            }
        }
    }


def test_get_registry_for_geometry_entity_info():
    """
    Test get_entity_info_and_registry_from_dict with GeometryEntityInfo.
    Uses geometry_grouped_by_file/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    params_as_dict, _ = SimulationParams._update_param_dict(params_as_dict)
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    _, registry = get_entity_info_and_registry_from_dict(params_as_dict)

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

    assert isinstance(registry, EntityRegistry)
    surfaces = registry.find_by_type(Surface)
    edges = registry.find_by_type(Edge)
    body_groups = registry.find_by_type(GeometryBodyGroup)
    generic_volumes = registry.find_by_type(GenericVolume)

    assert len(surfaces) == expected_surfaces_count
    assert len(edges) == expected_edges_count
    assert len(body_groups) == expected_bodies_count
    assert len(generic_volumes) == 0

    # Verify entity type names if entities exist
    if surfaces:
        assert surfaces[0].private_attribute_entity_type_name == "Surface"
    if edges:
        assert edges[0].private_attribute_entity_type_name == "Edge"
    if body_groups:
        assert body_groups[0].private_attribute_entity_type_name == "GeometryBodyGroup"


def test_get_registry_for_volume_mesh_entity_info():
    """
    Test get_entity_info_and_registry_from_dict with VolumeMeshEntityInfo.
    Uses vm_entity_provider/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("data/vm_entity_provider/simulation.json")
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    _, registry = get_entity_info_and_registry_from_dict(params_as_dict)

    # Get expected counts from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))
    expected_zones_count = len(entity_info.get("zones", []))

    assert isinstance(registry, EntityRegistry)
    surfaces = registry.find_by_type(Surface)
    generic_volumes = registry.find_by_type(GenericVolume)
    edges = registry.find_by_type(Edge)
    body_groups = registry.find_by_type(GeometryBodyGroup)

    assert len(surfaces) == expected_boundaries_count
    assert len(generic_volumes) == expected_zones_count
    assert len(edges) == 0
    assert len(body_groups) == 0

    # Verify entity type names if entities exist
    if surfaces:
        assert surfaces[0].private_attribute_entity_type_name == "Surface"
    if generic_volumes:
        assert generic_volumes[0].private_attribute_entity_type_name == "GenericVolume"


def test_get_registry_for_surface_mesh_entity_info():
    """
    Test get_entity_info_and_registry_from_dict with SurfaceMeshEntityInfo.
    Uses params/data/surface_mesh/simulation.json as test data.
    """
    params_as_dict = _load_simulation_json("params/data/surface_mesh/simulation.json")
    params_as_dict, _ = SimulationParams._update_param_dict(params_as_dict)
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    _, registry = get_entity_info_and_registry_from_dict(params_as_dict)

    # Get expected count from entity_info
    expected_boundaries_count = len(entity_info.get("boundaries", []))

    assert isinstance(registry, EntityRegistry)
    surfaces = registry.find_by_type(Surface)
    edges = registry.find_by_type(Edge)
    body_groups = registry.find_by_type(GeometryBodyGroup)
    generic_volumes = registry.find_by_type(GenericVolume)

    assert len(surfaces) == expected_boundaries_count
    assert len(edges) == 0
    assert len(body_groups) == 0
    assert len(generic_volumes) == 0

    # Verify entity type name if entities exist
    if surfaces:
        assert surfaces[0].private_attribute_entity_type_name == "Surface"


def test_get_registry_missing_asset_cache():
    """
    Test that the function raises ValueError when private_attribute_asset_cache is missing.
    """
    params_as_dict = {}

    with pytest.raises(ValueError, match="private_attribute_asset_cache not found"):
        get_entity_info_and_registry_from_dict(params_as_dict)


def test_get_registry_missing_entity_info():
    """
    Test that the function raises ValueError when project_entity_info is missing.
    """
    params_as_dict = {"private_attribute_asset_cache": {}}

    with pytest.raises(ValueError, match="project_entity_info not found"):
        get_entity_info_and_registry_from_dict(params_as_dict)


def test_geometry_entity_info_respects_grouping_tags():
    """
    Test that GeometryEntityInfo uses the correct grouping tags to extract entities.
    Verifies the function extracts entities based on the set grouping tag.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    params_as_dict, _ = SimulationParams._update_param_dict(params_as_dict)
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]
    _, registry = get_entity_info_and_registry_from_dict(params_as_dict)

    # Verify face grouping
    face_group_tag = entity_info.get("face_group_tag")
    assert face_group_tag is not None, "Test data should have face_group_tag set"

    face_attribute_names = entity_info.get("face_attribute_names", [])
    grouped_faces = entity_info.get("grouped_faces", [])
    index = face_attribute_names.index(face_group_tag)
    expected_faces = grouped_faces[index]

    surfaces = registry.find_by_type(Surface)
    assert len(surfaces) == len(expected_faces)

    # Verify edge grouping
    if entity_info.get("edge_group_tag"):
        edge_group_tag = entity_info["edge_group_tag"]
        edge_attribute_names = entity_info.get("edge_attribute_names", [])
        grouped_edges = entity_info.get("grouped_edges", [])
        index = edge_attribute_names.index(edge_group_tag)
        expected_edges = grouped_edges[index]
        edges = registry.find_by_type(Edge)
        assert len(edges) == len(expected_edges)

    # Verify body grouping
    if entity_info.get("body_group_tag"):
        body_group_tag = entity_info["body_group_tag"]
        body_attribute_names = entity_info.get("body_attribute_names", [])
        grouped_bodies = entity_info.get("grouped_bodies", [])
        index = body_attribute_names.index(body_group_tag)
        expected_bodies = grouped_bodies[index]
        body_groups = registry.find_by_type(GeometryBodyGroup)
        assert len(body_groups) == len(expected_bodies)


def test_get_registry_from_params_matches_dict():
    params_as_dict = _build_simple_params_dict()
    dummy_params = _DummyParams(params_as_dict)

    _, dict_registry = get_entity_info_and_registry_from_dict(params_as_dict)
    instance_registry = get_registry_from_params(dummy_params)

    assert isinstance(instance_registry, EntityRegistry)

    dict_surfaces = dict_registry.find_by_type(Surface)
    instance_surfaces = instance_registry.find_by_type(Surface)
    assert _entity_names(dict_surfaces) == _entity_names(instance_surfaces)

    dict_edges = dict_registry.find_by_type(Edge)
    instance_edges = instance_registry.find_by_type(Edge)
    assert _entity_names(dict_edges) == _entity_names(instance_edges)

    dict_body_groups = dict_registry.find_by_type(GeometryBodyGroup)
    instance_body_groups = instance_registry.find_by_type(GeometryBodyGroup)
    assert _entity_names(dict_body_groups) == _entity_names(instance_body_groups)

    dict_volumes = dict_registry.find_by_type(GenericVolume)
    instance_volumes = instance_registry.find_by_type(GenericVolume)
    assert _entity_names(dict_volumes) == _entity_names(instance_volumes)


def test_entity_registry_respects_grouping_selection():
    """
    Test that EntityRegistry.from_entity_info() only registers entities from the selected grouping.

    When GeometryEntityInfo has multiple groupings (e.g., by face, by body, all-in-one),
    the registry should only include entities from the grouping specified by face_group_tag,
    edge_group_tag, and body_group_tag.
    """
    params_as_dict = _load_simulation_json("data/geometry_grouped_by_file/simulation.json")
    params_as_dict, _ = SimulationParams._update_param_dict(params_as_dict)
    entity_info = params_as_dict["private_attribute_asset_cache"]["project_entity_info"]

    # Get the actual entity_info object by deserializing
    _, registry = get_entity_info_and_registry_from_dict(params_as_dict)

    # Verify we're testing GeometryEntityInfo with multiple groupings
    assert entity_info["type_name"] == "GeometryEntityInfo"
    assert (
        len(entity_info.get("face_attribute_names", [])) > 1
    ), "Test requires multiple face groupings"

    # Get the selected grouping index and expected entities
    face_group_tag = entity_info.get("face_group_tag")
    assert face_group_tag is not None, "Test requires face_group_tag to be set"

    face_attribute_names = entity_info["face_attribute_names"]
    grouped_faces = entity_info["grouped_faces"]

    selected_face_index = face_attribute_names.index(face_group_tag)
    expected_surface_names = [face["name"] for face in grouped_faces[selected_face_index]]

    # Get surfaces from registry
    registered_surfaces = registry.find_by_type(Surface)
    registered_surface_names = [surface.name for surface in registered_surfaces]

    # Verify ONLY surfaces from the selected grouping are registered
    assert set(registered_surface_names) == set(expected_surface_names), (
        f"Registry should only contain surfaces from grouping '{face_group_tag}' "
        f"(index {selected_face_index}), but got different entities"
    )

    # Verify surfaces from OTHER groupings are NOT registered
    for i, grouping_name in enumerate(face_attribute_names):
        if i != selected_face_index:
            other_grouping_names = [face["name"] for face in grouped_faces[i]]
            # Check that none of these names appear in registered surfaces
            overlap = set(other_grouping_names) & set(registered_surface_names)
            # Allow overlap only if the same surface name appears in multiple groupings
            # (which can happen in geometry files)
            for name in overlap:
                # Verify this is the same surface from the selected grouping
                assert name in expected_surface_names, (
                    f"Surface '{name}' from non-selected grouping '{grouping_name}' "
                    f"should not be registered when grouping tag is '{face_group_tag}'"
                )

    # Test edge grouping if available
    edge_group_tag = entity_info.get("edge_group_tag")
    if edge_group_tag and entity_info.get("edge_attribute_names"):
        edge_attribute_names = entity_info["edge_attribute_names"]
        grouped_edges = entity_info["grouped_edges"]

        if edge_group_tag in edge_attribute_names:
            selected_edge_index = edge_attribute_names.index(edge_group_tag)
            expected_edge_names = [edge["name"] for edge in grouped_edges[selected_edge_index]]

            registered_edges = registry.find_by_type(Edge)
            registered_edge_names = [edge.name for edge in registered_edges]

            assert set(registered_edge_names) == set(
                expected_edge_names
            ), f"Registry should only contain edges from grouping '{edge_group_tag}'"

    # Test body grouping if available
    body_group_tag = entity_info.get("body_group_tag")
    if body_group_tag and entity_info.get("body_attribute_names"):
        body_attribute_names = entity_info["body_attribute_names"]
        grouped_bodies = entity_info["grouped_bodies"]

        if body_group_tag in body_attribute_names:
            selected_body_index = body_attribute_names.index(body_group_tag)
            expected_body_names = [body["name"] for body in grouped_bodies[selected_body_index]]

            registered_bodies = registry.find_by_type(GeometryBodyGroup)
            registered_body_names = [body.name for body in registered_bodies]

            assert set(registered_body_names) == set(
                expected_body_names
            ), f"Registry should only contain bodies from grouping '{body_group_tag}'"
