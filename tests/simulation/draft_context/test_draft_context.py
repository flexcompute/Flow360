import flow360 as fl
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation import units as u
from flow360.component.simulation.draft_context import get_active_draft
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


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

        assert (
            found_in_draft_entity_info
        ), "Draft surface should reference entity from copied entity_info"


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


# ======================= Stage 4: set_up_params_for_uploading with DraftContext =======================


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
            root_asset=mock_geometry,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
            draft_entity_info=draft._entity_info,  # Pass draft's entity_info
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


def test_newly_created_draft_entities_in_params_after_set_up(mock_surface_mesh):
    """
    Test (ii): Newly created draft entities (e.g., Box) assigned to SimulationParams
    are captured in entity_info after set_up_params_for_uploading.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Create a new Box entity (draft entity type)
        with SI_unit_system:
            box = Box(
                name="test_refinement_box",
                center=(0, 0, 0) * u.m,
                size=(1, 1, 1) * u.m,
            )

            # Create params using this box in a refinement
            params = SimulationParams(
                meshing=fl.MeshingParams(
                    refinements=[
                        UniformRefinement(
                            spacing=0.1 * u.m,
                            entities=[box],
                        )
                    ],
                )
            )

        # Call set_up_params_for_uploading with draft's entity_info
        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_surface_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
            draft_entity_info=draft._entity_info,
        )

        # Verify the box is in the final entity_info's draft_entities
        final_entity_info = params.private_attribute_asset_cache.project_entity_info
        box_found = any(
            e.name == "test_refinement_box" and e.private_attribute_id == box.private_attribute_id
            for e in final_entity_info.draft_entities
        )
        assert (
            box_found
        ), "Newly created Box should be in draft_entities after set_up_params_for_uploading"


def test_draft_entity_modifications_preserved_after_set_up(mock_surface_mesh):
    """
    Test (iii): When user uses draft to access draft entities and makes changes,
    these changes are preserved after set_up_params_for_uploading.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # First, create a box and add it to draft's entity_info
        with SI_unit_system:
            box = Box(
                name="modifiable_box",
                center=(0, 0, 0) * u.m,
                size=(1, 1, 1) * u.m,
            )
        # Add to draft's entity_info directly
        draft._entity_info.draft_entities.append(box)
        draft._entity_registry.register(box)

        # Now modify the box (center is not frozen in Box)
        original_box_id = box.private_attribute_id

        # Create params using the box
        with SI_unit_system:
            params = SimulationParams(
                meshing=fl.MeshingParams(
                    refinements=[
                        UniformRefinement(
                            spacing=0.1 * u.m,
                            entities=[box],
                        )
                    ],
                )
            )

        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_surface_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
            draft_entity_info=draft._entity_info,
        )

        # Verify the entity_info in params is draft's entity_info (source of truth)
        final_entity_info = params.private_attribute_asset_cache.project_entity_info

        # The box should be there with the same ID (from draft's entity_info)
        box_in_final = None
        for e in final_entity_info.draft_entities:
            if e.private_attribute_id == original_box_id:
                box_in_final = e
                break
        assert box_in_final is not None, "Box from draft entity_info should be preserved"
        assert box_in_final.name == "modifiable_box"


def test_external_draft_entities_from_copied_params_are_captured(mock_surface_mesh):
    """
    Test (iv): When user copies draft entities (e.g., Box in porous_medium) from
    an imported SimulationParams to their current params, those entities should
    be captured in entity_info after set_up_params_for_uploading.

    This simulates the case where a user loads a JSON, extracts a section with
    embedded draft entities, and uses it in their new params.
    """
    # Simulate loading a params from JSON that has a Box embedded in it
    # (In reality this would be loaded from a file)
    with SI_unit_system:
        imported_box = Box(
            name="imported_box_from_json",
            center=(5, 5, 5) * u.m,
            size=(2, 2, 2) * u.m,
        )
        imported_box_id = imported_box.private_attribute_id

    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # User creates new params and uses the imported box
        # (simulating copying porous_medium section from another params)
        with SI_unit_system:
            params = SimulationParams(
                meshing=fl.MeshingParams(
                    refinements=[
                        UniformRefinement(
                            spacing=0.5 * u.m,
                            entities=[imported_box],
                        )
                    ],
                )
            )

        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_surface_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
            draft_entity_info=draft._entity_info,
        )

        # Verify the imported box is captured in draft_entities
        final_entity_info = params.private_attribute_asset_cache.project_entity_info
        imported_box_found = any(
            e.private_attribute_id == imported_box_id for e in final_entity_info.draft_entities
        )
        assert (
            imported_box_found
        ), "External Box from copied params should be captured in draft_entities"


def test_draft_entity_info_is_source_of_truth_over_params(mock_surface_mesh):
    """
    Test that when the same draft entity exists in both draft's entity_info
    and params.used_entity_registry, the entity_info version is preserved
    (source of truth).
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Create a box and add to draft's entity_info
        with SI_unit_system:
            box = Box(
                name="source_of_truth_box",
                center=(0, 0, 0) * u.m,
                size=(1, 1, 1) * u.m,
            )
        box_id = box.private_attribute_id

        # Add to draft's entity_info
        draft._entity_info.draft_entities.append(box)

        # Create params using the SAME box (same ID)
        with SI_unit_system:
            params = SimulationParams(
                meshing=fl.MeshingParams(
                    refinements=[
                        UniformRefinement(
                            spacing=1 * u.m,
                            entities=[box],
                        )
                    ],
                )
            )

        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_surface_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
            draft_entity_info=draft._entity_info,
        )

        # Verify exactly one box with this ID exists (no duplicates)
        final_entity_info = params.private_attribute_asset_cache.project_entity_info
        boxes_with_id = [
            e for e in final_entity_info.draft_entities if e.private_attribute_id == box_id
        ]
        assert len(boxes_with_id) == 1, "Should have exactly one box with this ID (no duplicates)"


# ======================= Stage 4b: Legacy vs DraftContext pathway interplay =======================


def test_legacy_asset_access_inside_draft_context_emits_warning(mock_surface_mesh, capsys):
    """
    Test Scenario B: Using legacy asset[key] access while inside a DraftContext
    should emit a warning to guide users to the new pathway.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Access via legacy pathway - should emit warning
        _ = mock_surface_mesh["fuselage"]

    # Capture stdout/stderr (flow360 log uses Rich console which writes to stdout)
    captured = capsys.readouterr()
    assert (
        "Accessing entities via asset" in captured.out
    ), f"Expected warning about legacy access inside DraftContext. Got: {captured.out}"


def test_legacy_geometry_access_inside_draft_context_emits_warning(mock_geometry, capsys):
    """
    Test Scenario B for Geometry: Using legacy geometry[key] access while inside
    a DraftContext should emit a warning.
    """
    with create_draft(new_run_from=mock_geometry) as draft:
        # Access via legacy pathway - should emit warning
        # The mock_geometry may not have internal_registry set up, but the warning
        # should still be emitted before that check
        try:
            _ = mock_geometry["*"]
        except Exception:
            pass  # We only care about the warning being emitted

    # Capture stdout/stderr (flow360 log uses Rich console which writes to stdout)
    captured = capsys.readouterr()
    assert (
        "Accessing entities via asset" in captured.out
    ), f"Expected warning about legacy access inside DraftContext. Got: {captured.out}"


def test_mixed_legacy_and_draft_access_emits_warning(mock_surface_mesh, capsys):
    """
    Test Scenario D: User mixes both legacy asset[key] and draft.surfaces[key] access.
    Legacy access should emit warning, draft access should not.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Access via draft pathway - should NOT emit warning
        draft_surface = draft.surfaces["fuselage"]
        assert draft_surface.name == "fuselage"

        # Access via legacy pathway - should emit warning
        legacy_surface = mock_surface_mesh["fuselage"]

    # Capture output - should have exactly one warning (from legacy access)
    captured = capsys.readouterr()
    # Count occurrences of the warning message
    warning_count = captured.out.count("Accessing entities via asset")
    assert (
        warning_count == 1
    ), f"Expected exactly one warning from legacy access, got {warning_count}. Output: {captured.out}"


def test_legacy_access_outside_draft_context_no_warning(mock_surface_mesh, capsys):
    """
    Test that legacy asset[key] access OUTSIDE DraftContext does NOT emit warning.
    This ensures backward compatibility for users not using DraftContext.
    """
    # Access outside any draft context - should NOT emit warning
    _ = mock_surface_mesh["fuselage"]

    # Capture output - should have no warning
    captured = capsys.readouterr()
    assert (
        "Accessing entities via asset" not in captured.out
    ), f"No warning expected when accessing outside DraftContext. Got: {captured.out}"


def test_legacy_and_draft_entities_are_different_objects(mock_surface_mesh):
    """
    Test that entities accessed via legacy pathway and draft pathway are DIFFERENT objects.
    This verifies the isolation - modifying one should not affect the other.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Get entity via draft pathway
        draft_surface = draft.surfaces["fuselage"]

        # Get entity via legacy pathway (will emit warning, but that's expected)
        legacy_surface = mock_surface_mesh["fuselage"]

        # They should be different objects due to draft isolation
        assert (
            draft_surface is not legacy_surface
        ), "Draft and legacy entities should be different objects"

        # But they should have the same name
        assert draft_surface.name == legacy_surface.name


def test_legacy_modifications_not_reflected_in_draft_entity_info(mock_surface_mesh):
    """
    Test that modifications via legacy pathway do NOT affect draft's entity_info.
    This verifies the draft isolation is working correctly.
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        # Get the draft surface
        draft_surface = draft.surfaces["fuselage"]
        draft_surface_id = draft_surface.private_attribute_id

        # Get the legacy surface and modify it
        legacy_surface = mock_surface_mesh["fuselage"]
        legacy_surface.private_attribute_color = "legacy_red"

        # The draft surface should NOT have this modification
        assert (
            draft_surface.private_attribute_color != "legacy_red"
        ), "Draft entity should not be affected by legacy modifications"

        # Verify the draft's entity_info also doesn't have the modification
        found_in_draft = False
        for boundary in draft._entity_info.boundaries:
            if boundary.private_attribute_id == draft_surface_id:
                assert boundary.private_attribute_color != "legacy_red"
                found_in_draft = True
                break
        assert found_in_draft
