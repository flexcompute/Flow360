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


def test_create_draft_accepts_geometry_grouping_override(mock_geometry):
    assert mock_geometry.entity_info.face_group_tag == "ByBody"
    with create_draft(new_run_from=mock_geometry, face_grouping="faceId") as draft:
        assert draft._entity_info.face_group_tag == "faceId"


# ======================= Draft Entity Isolation =======================


def test_draft_entity_info_is_deep_copy(mock_surface_mesh):
    """Test that DraftContext receives a deep copy of entity_info, not a reference."""
    original_entity_info = mock_surface_mesh.entity_info

    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # The draft's entity_info should be a different object
        assert draft._entity_info is not original_entity_info

        # But they should have the same type
        assert type(draft._entity_info) is type(original_entity_info)

        # And the same content (initially)
        assert draft._entity_info.type_name == original_entity_info.type_name


def test_draft_entity_modifications_are_isolated(mock_surface_mesh):
    """Test that modifications in draft don't affect the original asset's entity_info."""
    # Get the original surface name
    original_entity_info = mock_surface_mesh.entity_info
    original_boundaries = list(original_entity_info.boundaries)
    assert len(original_boundaries) > 0

    # Get the original name of the first boundary
    original_name = original_boundaries[0].name

    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Get the first surface from the draft
        draft_surface = list(draft.surfaces)[0]

        # The draft surface should have the same name initially
        assert draft_surface.name == original_name

        # The draft surface should be a DIFFERENT object than the original
        original_surface = original_boundaries[0]
        assert draft_surface is not original_surface

    # After exiting draft, original entity_info should be unchanged
    assert original_boundaries[0].name == original_name


def test_draft_entity_info_is_independent_for_geometry(mock_geometry):
    """Test draft isolation works for geometry assets."""
    original_entity_info = mock_geometry.entity_info

    with create_draft(new_run_from=mock_geometry) as draft:
        # The draft's entity_info should be a different object
        assert draft._entity_info is not original_entity_info

        # Verify the type
        assert draft._entity_info.type_name == "GeometryEntityInfo"

        # Get surfaces from draft - they should be different objects
        if len(list(draft.surfaces)) > 0:
            draft_surface = list(draft.surfaces)[0]
            # Find corresponding surface in original
            for group in original_entity_info.grouped_faces:
                for surface in group:
                    if surface.name == draft_surface.name:
                        # Same name but different object
                        assert draft_surface is not surface
                        break


def test_draft_entities_reference_copied_entity_info(mock_surface_mesh):
    """Test that entities in draft registry reference the copied entity_info, not original."""
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Get a surface from the draft
        draft_surface = draft.surfaces["fuselage"]

        # This surface should be in the draft's entity_info.boundaries
        found_in_draft_entity_info = False
        for boundary in draft._entity_info.boundaries:
            if boundary is draft_surface:
                found_in_draft_entity_info = True
                break

        assert found_in_draft_entity_info, "Draft surface should reference entity from copied entity_info"


def test_multiple_drafts_are_isolated_from_each_other(mock_surface_mesh):
    """Test that multiple drafts created from the same asset are isolated."""
    # Create first draft and get entity IDs
    with create_draft(new_run_from=mock_surface_mesh) as draft1:
        draft1_surfaces = list(draft1.surfaces)
        draft1_entity_info = draft1._entity_info

    # Create second draft
    with create_draft(new_run_from=mock_surface_mesh) as draft2:
        draft2_surfaces = list(draft2.surfaces)
        draft2_entity_info = draft2._entity_info

        # The entity_info objects should be different
        assert draft1_entity_info is not draft2_entity_info

        # Surfaces should be different objects (even if same names)
        for s1 in draft1_surfaces:
            for s2 in draft2_surfaces:
                if s1.name == s2.name:
                    assert s1 is not s2


def test_draft_uses_entity_registry_from_entity_info(mock_surface_mesh):
    """Test that DraftContext uses EntityRegistry.from_entity_info() for building registry."""
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # The registry should be populated with entities
        assert draft._entity_registry.entity_count() > 0

        # All surfaces in the registry should be the same objects as in entity_info
        for surface in draft.surfaces:
            found = False
            for boundary in draft._entity_info.boundaries:
                if boundary is surface:
                    found = True
                    break
            assert found, f"Surface {surface.name} should be same object as in entity_info"


# TODO: Re enable the test once we are all done.
# def test_draft_entity_modifications_flow_to_params_without_update_persistent_entities(
#     mock_geometry,
# ):
#     """
#     Test: Entity modifications via draft context flow through to params
#     WITHOUT needing update_persistent_entities().

#     This test verifies that the draft context approach achieves the same behavior
#     as the legacy update_persistent_entities() mechanism, but with direct reference
#     identity instead of registry sync.

#     The legacy flow was:
#     1. User modifies entity via asset["wing"].color = "red"
#     2. Modification stored in asset.internal_registry
#     3. update_persistent_entities() syncs registry back to entity_info
#     4. entity_info is used in params

#     The draft flow is:
#     1. User modifies entity via draft.surfaces["wing"].color = "red"
#     2. Modification is DIRECTLY on the entity_info entity (same object)
#     3. No sync needed - entity_info already has the change
#     4. entity_info is used in params
#     """
#     # Use default grouping to keep entities consistent between draft and entity_info
#     with create_draft(new_run_from=mock_geometry) as draft:
#         # Get a surface from the draft and modify it
#         # The surfaces in draft are the SAME objects as in entity_info
#         surface = draft.surfaces["*"][0]
#         surface_id = surface.private_attribute_id

#         # Modify a non-frozen attribute (private_attribute_color is not frozen)
#         surface.private_attribute_color = "test_red_color"

#         # Verify the change is immediately reflected in entity_info
#         # Find the same entity by ID in entity_info
#         entity_in_info = None
#         for group in draft._entity_info.grouped_faces:
#             for entity in group:
#                 if entity.private_attribute_id == surface_id:
#                     entity_in_info = entity
#                     break
#         assert entity_in_info is not None, "Entity not found in entity_info"
#         assert entity_in_info.private_attribute_color == "test_red_color"
#         assert entity_in_info is surface  # Same object!

#         # Now create params and call set_up_params_for_uploading
#         # Use a specific surface with consistent tag to avoid grouping conflicts
#         with fl.SI_unit_system:
#             params = fl.SimulationParams(
#                 outputs=[fl.SurfaceOutput(surfaces=[surface], output_fields=["Cp"])],
#             )

#         params = set_up_params_for_uploading(
#             params=params,
#             root_asset=mock_geometry,  # root_asset is passed but should be ignored in draft mode
#             length_unit=1 * u.m,
#             use_beta_mesher=False,
#             use_geometry_AI=False,
#         )

#         # Verify the modification made it through to the final params
#         # Find the entity by ID in the final params
#         final_entity = None
#         for group in params.private_attribute_asset_cache.project_entity_info.grouped_faces:
#             for entity in group:
#                 if entity.private_attribute_id == surface_id:
#                     final_entity = entity
#                     break
#         assert final_entity is not None, "Entity not found in final params"
#         assert final_entity.private_attribute_color == "test_red_color"
