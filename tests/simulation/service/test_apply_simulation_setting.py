"""Tests for apply_simulation_setting_to_entity_info service function."""

import copy

import pytest

from flow360.component.simulation import services


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def _create_base_simulation_dict(entity_info_type="VolumeMeshEntityInfo", surfaces=None):
    """Helper to create a base simulation dict with configurable entity info."""
    if surfaces is None:
        surfaces = []

    entity_info = {
        "draft_entities": [],
        "ghost_entities": [],
        "type_name": entity_info_type,
    }

    if entity_info_type == "VolumeMeshEntityInfo":
        entity_info["zones"] = []
        entity_info["boundaries"] = surfaces
    elif entity_info_type == "SurfaceMeshEntityInfo":
        entity_info["boundaries"] = surfaces
    elif entity_info_type == "GeometryEntityInfo":
        entity_info["grouped_faces"] = [surfaces] if surfaces else [[]]
        entity_info["grouped_edges"] = [[]]
        entity_info["grouped_bodies"] = [[]]
        entity_info["face_attribute_names"] = ["default"]
        entity_info["edge_attribute_names"] = ["default"]
        entity_info["body_attribute_names"] = ["default"]
        entity_info["face_group_tag"] = "default"
        entity_info["edge_group_tag"] = "default"
        entity_info["body_group_tag"] = "default"
        entity_info["body_ids"] = []
        entity_info["face_ids"] = []
        entity_info["edge_ids"] = []

    return {
        "meshing": {
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "defaults": {
                "surface_edge_growth_rate": 1.5,
                "boundary_layer_first_layer_thickness": "1*m",
                "surface_max_edge_length": "1*m",
            },
            "refinements": [],
            "volume_zones": [],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": 1.0, "units": "m**2"},
        },
        "time_stepping": {
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "models": [],
        "outputs": [],
        "user_defined_dynamics": [],
        "unit_system": {"name": "SI"},
        "version": "24.11.0",
        "private_attribute_asset_cache": {
            "project_length_unit": None,
            "project_entity_info": entity_info,
            "use_inhouse_mesher": True,
            "use_geometry_AI": False,
        },
    }


def _create_surface_entity(name, private_attribute_id=None):
    """Helper to create a surface entity dict."""
    return {
        "private_attribute_registry_bucket_name": "SurfaceEntityType",
        "private_attribute_entity_type_name": "Surface",
        "name": name,
        "private_attribute_id": private_attribute_id or name,
        "private_attribute_is_interface": False,
        "private_attribute_sub_components": [],
    }


def _create_box_entity(name, center=None):
    """Helper to create a box (draft) entity dict."""
    return {
        "private_attribute_registry_bucket_name": "VolumetricEntityType",
        "private_attribute_entity_type_name": "Box",
        "type_name": "Box",
        "name": name,
        "center": center or {"value": [0, 0, 0], "units": "m"},
        "size": {"value": [1, 1, 1], "units": "m"},
        "axis_of_rotation": [0, 0, 1],
        "angle_of_rotation": {"value": 0, "units": "degree"},
    }


class TestApplySimulationSettingBasic:
    """Basic tests for apply_simulation_setting_to_entity_info."""

    def test_basic_entity_replacement(self):
        """Test that entities are correctly replaced when names match."""
        # Source simulation with settings and entity "wing"
        source_surface = _create_surface_entity("wing", "source_wing_id")
        source_dict = _create_base_simulation_dict(surfaces=[source_surface])
        source_dict["models"] = [
            {
                "type": "Wall",
                "entities": {"stored_entities": [_create_surface_entity("wing", "source_wing_id")]},
                "use_wall_function": False,
            }
        ]

        # Target simulation with different entity info (same name "wing" but different id)
        target_surface = _create_surface_entity("wing", "target_wing_id")
        target_dict = _create_base_simulation_dict(surfaces=[target_surface])

        # Apply settings from source to target's entity info
        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: entity in models should now reference target's entity
        stored_entity = result_dict["models"][0]["entities"]["stored_entities"][0]
        assert stored_entity["private_attribute_id"] == "target_wing_id"
        assert stored_entity["name"] == "wing"

        # Verify: project_entity_info should be from target
        boundary = result_dict["private_attribute_asset_cache"]["project_entity_info"][
            "boundaries"
        ][0]
        assert boundary["private_attribute_id"] == "target_wing_id"


class TestApplySimulationSettingUnmatchedEntity:
    """Tests for unmatched entity handling."""

    def test_unmatched_entity_generates_warning(self):
        """Test that unmatched entities are skipped and generate warnings."""
        # Source has entity "wing" and "fuselage"
        source_surfaces = [
            _create_surface_entity("wing", "wing_id"),
            _create_surface_entity("fuselage", "fuselage_id"),
        ]
        source_dict = _create_base_simulation_dict(surfaces=source_surfaces)
        source_dict["models"] = [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        _create_surface_entity("wing", "wing_id"),
                        _create_surface_entity("fuselage", "fuselage_id"),
                    ]
                },
                "use_wall_function": False,
            }
        ]

        # Target only has "wing", no "fuselage"
        target_surfaces = [_create_surface_entity("wing", "target_wing_id")]
        target_dict = _create_base_simulation_dict(surfaces=target_surfaces)

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: only "wing" should remain in stored_entities
        stored = result_dict["models"][0]["entities"]["stored_entities"]
        assert len(stored) == 1
        assert stored[0]["name"] == "wing"

        # Verify: warning generated for "fuselage"
        entity_warnings = [w for w in warnings if w.get("type") == "entity_not_found"]
        assert len(entity_warnings) == 1
        assert "fuselage" in entity_warnings[0]["msg"]


class TestApplySimulationSettingPreserveDraftEntities:
    """Tests for draft/ghost entity preservation."""

    def test_draft_entities_preserved_in_entity_info(self):
        """Test that draft_entities from source are preserved in result."""
        # Source has a box in draft_entities
        source_dict = _create_base_simulation_dict()
        source_dict["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"] = [
            _create_box_entity("refinement_box")
        ]

        # Target has no draft entities
        target_dict = _create_base_simulation_dict()
        target_dict["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"] = []

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: draft_entities from source should be preserved
        draft_entities = result_dict["private_attribute_asset_cache"]["project_entity_info"][
            "draft_entities"
        ]
        assert len(draft_entities) == 1
        assert draft_entities[0]["name"] == "refinement_box"

    def test_draft_entities_in_stored_entities_preserved(self):
        """Test that draft entities in stored_entities are preserved without matching."""
        # Source with a Box entity in stored_entities (used in refinement)
        source_dict = _create_base_simulation_dict()
        source_dict["meshing"]["refinements"] = [
            {
                "type": "UniformRefinement",
                "entities": {"stored_entities": [_create_box_entity("my_box")]},
                "spacing": {"value": 0.1, "units": "m"},
            }
        ]

        # Target with different entity info
        target_dict = _create_base_simulation_dict()

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: Box entity should be preserved (not matched/removed)
        stored = result_dict["meshing"]["refinements"][0]["entities"]["stored_entities"]
        assert len(stored) == 1
        assert stored[0]["name"] == "my_box"
        assert stored[0]["private_attribute_entity_type_name"] == "Box"

        # Verify: no warning for Box entity
        entity_warnings = [w for w in warnings if w.get("type") == "entity_not_found"]
        assert len(entity_warnings) == 0


class TestApplySimulationSettingCrossEntityInfoType:
    """Tests for applying settings across different entity info types."""

    def test_volume_mesh_to_volume_mesh(self):
        """Test applying settings from one VolumeMesh project to another."""
        source_surface = _create_surface_entity("inlet", "source_inlet_id")
        source_dict = _create_base_simulation_dict(surfaces=[source_surface])
        source_dict["models"] = [
            {
                "type": "Freestream",
                "entities": {
                    "stored_entities": [_create_surface_entity("inlet", "source_inlet_id")]
                },
            }
        ]

        target_surface = _create_surface_entity("inlet", "target_inlet_id")
        target_dict = _create_base_simulation_dict(surfaces=[target_surface])

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: entity replaced with target's
        stored = result_dict["models"][0]["entities"]["stored_entities"]
        assert stored[0]["private_attribute_id"] == "target_inlet_id"

    def test_geometry_to_volume_mesh_no_grouping_tags(self):
        """Test that grouping tags from Geometry source don't leak into VolumeMesh target."""
        # Source is GeometryEntityInfo with grouping tags
        source_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "some_grouping"

        # Target is VolumeMeshEntityInfo (no grouping tags)
        target_surface = _create_surface_entity("wall", "target_wall_id")
        target_dict = _create_base_simulation_dict(
            entity_info_type="VolumeMeshEntityInfo", surfaces=[target_surface]
        )

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: no grouping tags should be in result entity_info
        result_entity_info = result_dict["private_attribute_asset_cache"]["project_entity_info"]
        assert "face_group_tag" not in result_entity_info
        assert "body_group_tag" not in result_entity_info
        assert "edge_group_tag" not in result_entity_info

        # Verify: result type should still be VolumeMeshEntityInfo
        assert result_entity_info["type_name"] == "VolumeMeshEntityInfo"


class TestApplySimulationSettingGroupingTags:
    """Tests for grouping tag inheritance from source."""

    def test_grouping_tags_inherited_from_source_when_available_in_target(self):
        """Test that grouping tags are inherited from source when they exist in target's attribute_names."""
        # Source uses "groupA" for all groupings
        source_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupA"
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "body_group_tag"
        ] = "groupA"
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "edge_group_tag"
        ] = "groupA"
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = ["groupA", "groupB"]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "body_attribute_names"
        ] = ["groupA", "groupB"]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "edge_attribute_names"
        ] = ["groupA", "groupB"]

        # Target has both groupA and groupB, but uses groupB by default
        target_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupB"
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "body_group_tag"
        ] = "groupB"
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "edge_group_tag"
        ] = "groupB"
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = [
            "groupA",
            "groupB",
        ]  # Target has groupA available
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "body_attribute_names"
        ] = ["groupA", "groupB"]
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "edge_attribute_names"
        ] = ["groupA", "groupB"]

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: grouping tags should be from source since "groupA" exists in target's attribute_names
        result_entity_info = result_dict["private_attribute_asset_cache"]["project_entity_info"]
        assert result_entity_info["face_group_tag"] == "groupA"
        assert result_entity_info["body_group_tag"] == "groupA"
        assert result_entity_info["edge_group_tag"] == "groupA"

    def test_grouping_tags_use_target_when_source_is_none(self):
        """Test that target's grouping tags are used when source has None."""
        # Source with None grouping tags
        source_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        source_dict["private_attribute_asset_cache"]["project_entity_info"]["face_group_tag"] = None

        # Target with specific grouping tag
        target_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "target_face_grouping"

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: face_group_tag should remain from target since source is None
        result_entity_info = result_dict["private_attribute_asset_cache"]["project_entity_info"]
        assert result_entity_info["face_group_tag"] == "target_face_grouping"

    def test_geometry_entities_matched_with_source_grouping(self):
        """Test that entities are matched using source's grouping selection."""
        # Create surfaces for two different groupings
        wing_surface_group_a = _create_surface_entity("wing", "wing_group_a_id")
        wing_surface_group_b = _create_surface_entity("wing", "wing_group_b_id")

        # Source uses "groupA" for face grouping
        source_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        source_dict["private_attribute_asset_cache"]["project_entity_info"]["grouped_faces"] = [
            [wing_surface_group_a],  # groupA surfaces
            [],  # groupB surfaces (empty in source)
        ]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = ["groupA", "groupB"]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupA"
        source_dict["models"] = [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [_create_surface_entity("wing", "wing_group_a_id")]
                },
                "use_wall_function": False,
            }
        ]

        # Target has surfaces in both groupings
        target_wing_group_a = _create_surface_entity("wing", "target_wing_group_a_id")
        target_wing_group_b = _create_surface_entity("wing", "target_wing_group_b_id")
        target_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        target_dict["private_attribute_asset_cache"]["project_entity_info"]["grouped_faces"] = [
            [target_wing_group_a],  # groupA surfaces
            [target_wing_group_b],  # groupB surfaces
        ]
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = ["groupA", "groupB"]
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupB"  # Target originally uses groupB

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: result should use source's grouping (groupA), so entity should match groupA's wing
        stored = result_dict["models"][0]["entities"]["stored_entities"]
        assert len(stored) == 1
        assert stored[0]["private_attribute_id"] == "target_wing_group_a_id"

        # Verify: face_group_tag in result should be from source
        result_entity_info = result_dict["private_attribute_asset_cache"]["project_entity_info"]
        assert result_entity_info["face_group_tag"] == "groupA"

    def test_source_grouping_tag_not_in_target_falls_back_to_target_tag(self):
        """Test that when source's grouping tag doesn't exist in target, target's tag is used."""
        # Source uses "groupA" which target doesn't have
        source_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        source_dict["private_attribute_asset_cache"]["project_entity_info"]["grouped_faces"] = [
            [_create_surface_entity("wing", "source_wing_id")],
        ]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = ["groupA"]
        source_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupA"
        source_dict["models"] = [
            {
                "type": "Wall",
                "entities": {"stored_entities": [_create_surface_entity("wing", "source_wing_id")]},
                "use_wall_function": False,
            }
        ]

        # Target only has "groupB" (no "groupA")
        target_dict = _create_base_simulation_dict(entity_info_type="GeometryEntityInfo")
        target_dict["private_attribute_asset_cache"]["project_entity_info"]["grouped_faces"] = [
            [_create_surface_entity("wing", "target_wing_id")],
        ]
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_attribute_names"
        ] = ["groupB"]
        target_dict["private_attribute_asset_cache"]["project_entity_info"][
            "face_group_tag"
        ] = "groupB"

        result_dict, errors, warnings = services.apply_simulation_setting_to_entity_info(
            simulation_setting_dict=copy.deepcopy(source_dict),
            entity_info_dict=copy.deepcopy(target_dict),
        )

        # Verify: face_group_tag should fall back to target's "groupB" since "groupA" doesn't exist
        result_entity_info = result_dict["private_attribute_asset_cache"]["project_entity_info"]
        assert result_entity_info["face_group_tag"] == "groupB"

        # Verify: entity should still be matched using target's grouping
        stored = result_dict["models"][0]["entities"]["stored_entities"]
        assert len(stored) == 1
        assert stored[0]["private_attribute_id"] == "target_wing_id"
