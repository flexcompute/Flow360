import flow360 as fl
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation import units as u
from flow360.component.simulation.draft_context import get_active_draft


def test_create_draft_exposes_entity_registry(mock_surface_mesh):
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        assert get_active_draft() is draft
        fuselage = draft.surfaces["fuselage"]
        assert fuselage.name == "fuselage"

        tails = draft.surfaces["*tail"]
        tail_names = sorted([entity.name for entity in tails])
        assert tail_names == ["horizontal tail", "vertical tail"]

    assert get_active_draft() is None


def test_capture_changes_into_draft_registry_are_reflected_in_entity_info(mock_volume_mesh):
    with create_draft(new_run_from=mock_volume_mesh) as draft:
        draft.volumes["blk-1"].center = (1, 5, 3) * u.cm
        assert all(draft._entity_info.zones[0].center == (1, 5, 3) * u.cm)


def test_create_draft_accepts_geometry_grouping_override(mock_geometry):
    with create_draft(new_run_from=mock_geometry, face_grouping="faceId") as draft:
        assert draft._entity_info.face_group_tag == "faceId"


def test_draft_entity_modifications_flow_to_params_without_update_persistent_entities(
    mock_geometry,
):
    """
    Test: Entity modifications via draft context flow through to params
    WITHOUT needing update_persistent_entities().

    This test verifies that the draft context approach achieves the same behavior
    as the legacy update_persistent_entities() mechanism, but with direct reference
    identity instead of registry sync.

    The legacy flow was:
    1. User modifies entity via asset["wing"].color = "red"
    2. Modification stored in asset.internal_registry
    3. update_persistent_entities() syncs registry back to entity_info
    4. entity_info is used in params

    The draft flow is:
    1. User modifies entity via draft.surfaces["wing"].color = "red"
    2. Modification is DIRECTLY on the entity_info entity (same object)
    3. No sync needed - entity_info already has the change
    4. entity_info is used in params
    """
    # Use default grouping to keep entities consistent between draft and entity_info
    with create_draft(new_run_from=mock_geometry) as draft:
        # Get a surface from the draft and modify it
        # The surfaces in draft are the SAME objects as in entity_info
        surface = draft.surfaces["*"][0]
        surface_id = surface.private_attribute_id

        # Modify a non-frozen attribute (private_attribute_color is not frozen)
        surface.private_attribute_color = "test_red_color"

        # Verify the change is immediately reflected in entity_info
        # Find the same entity by ID in entity_info
        entity_in_info = None
        for group in draft._entity_info.grouped_faces:
            for entity in group:
                if entity.private_attribute_id == surface_id:
                    entity_in_info = entity
                    break
        assert entity_in_info is not None, "Entity not found in entity_info"
        assert entity_in_info.private_attribute_color == "test_red_color"
        assert entity_in_info is surface  # Same object!

        # Now create params and call set_up_params_for_uploading
        # Use a specific surface with consistent tag to avoid grouping conflicts
        with fl.SI_unit_system:
            params = fl.SimulationParams(
                outputs=[fl.SurfaceOutput(surfaces=[surface], output_fields=["Cp"])],
            )

        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_geometry,  # root_asset is passed but should be ignored in draft mode
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

        # Verify the modification made it through to the final params
        # Find the entity by ID in the final params
        final_entity = None
        for group in params.private_attribute_asset_cache.project_entity_info.grouped_faces:
            for entity in group:
                if entity.private_attribute_id == surface_id:
                    final_entity = entity
                    break
        assert final_entity is not None, "Entity not found in final params"
        assert final_entity.private_attribute_color == "test_red_color"


def test_draft_volume_zone_modifications_flow_to_params(mock_volume_mesh):
    """
    Test: Volume zone modifications via draft context flow through to params.

    This mirrors test_persistent_entity_info_update_volume_mesh but uses
    draft context instead of update_persistent_entities().
    """
    with create_draft(new_run_from=mock_volume_mesh) as draft:
        # Modify the center of a zone
        zone = draft.volumes["blk-1"]
        zone.center = (1.2, 2.3, 3.4) * u.cm

        # Verify change is in entity_info
        assert all(draft._entity_info.zones[0].center == (1.2, 2.3, 3.4) * u.cm)

        # Create params and go through set_up_params_for_uploading
        with fl.SI_unit_system:
            params = fl.SimulationParams()

        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_volume_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

        # Verify the modification made it through
        final_zone = params.private_attribute_asset_cache.project_entity_info.zones[0]
        assert all(final_zone.center == (1.2, 2.3, 3.4) * u.cm)
