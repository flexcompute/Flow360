"""Test that entity type filtering works correctly during selector expansion.

This test verifies the key behavior of the centralized field validator:
- Selectors can match entity types beyond what the EntityList accepts
- Field validator silently filters out invalid types during expansion
- No error is raised when expanded entities include invalid types
"""

import json
import os

import pytest

from flow360.component.simulation.draft_context import DraftContext
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import (
    SurfaceSelector,
    expand_entity_list_selectors_in_place,
)
from flow360.component.simulation.primitives import (
    GenericVolume,
    GhostSphere,
    MirroredSurface,
    Surface,
)


def test_selector_expansion_filters_invalid_types_silently():
    """
    Test that selector expansion with type expansion map works correctly.

    Scenario:
    - EntityList[Surface, MirroredSurface] only accepts Surface and MirroredSurface
    - SurfaceSelector with expansion map matches Surface, MirroredSurface, AND GhostSphere
    - Field validator should silently filter out GhostSphere
    - No error should be raised
    """
    # Create registry with mixed surface types
    registry = EntityRegistry()
    surface1 = Surface(name="wing")
    surface2 = MirroredSurface(
        name="wing_mirrored", surface_id="surface1", mirror_plane_id="plane1"
    )
    # GhostSphere is in the expansion map for Surface but not in EntityList[Surface, MirroredSurface]
    ghost_sphere = GhostSphere(name="ghost_sphere")

    registry.register(surface1)
    registry.register(surface2)
    registry.register(ghost_sphere)

    print("\nRegistry contains:")
    print(f"  - Surface: {surface1.name}")
    print(f"  - MirroredSurface: {surface2.name}")
    print(
        f"  - GhostSphere: {ghost_sphere.name} (in expansion map but not in EntityList valid types)"
    )

    # Create EntityList[Surface, MirroredSurface] with selector matching all
    selector = SurfaceSelector(name="all_surfaces").match("*")
    entity_list = EntityList[Surface, MirroredSurface](
        stored_entities=[],
        selectors=[selector],
    )

    print(f"\nEntityList valid types: Surface, MirroredSurface")
    print(f"Selector will match: Surface, MirroredSurface, GhostSphere (via expansion map)")

    # Expand selectors - should NOT raise error
    print(f"\nExpanding selector...")
    expand_entity_list_selectors_in_place(registry, entity_list)

    print(f"\nAfter expansion:")
    print(f"  stored_entities: {[e.name for e in entity_list.stored_entities]}")
    print(f"  Types: {[type(e).__name__ for e in entity_list.stored_entities]}")

    # Verify results
    entity_names = {e.name for e in entity_list.stored_entities}
    entity_types = {type(e).__name__ for e in entity_list.stored_entities}

    # Should include valid types
    assert "wing" in entity_names, "Surface should be included"
    assert "wing_mirrored" in entity_names, "MirroredSurface should be included"

    # Should NOT include invalid type
    assert "ghost_sphere" not in entity_names, "GhostSphere should be filtered out"

    # Verify only valid types present
    assert entity_types == {
        "Surface",
        "MirroredSurface",
    }, f"Only Surface and MirroredSurface should be present, got {entity_types}"

    print(f"\n✓ Test passed: Invalid type (GhostSphere) was silently filtered out")


def test_preview_selector_matches_mirrored_entities():
    """Test that DraftContext.preview_selector can match mirrored entities."""
    registry = EntityRegistry()
    surface = Surface(name="wing")
    mirrored_surface = MirroredSurface(
        name="wing_mirrored", surface_id="surface1", mirror_plane_id="plane1"
    )

    registry.register(surface)
    registry.register(mirrored_surface)

    # Build a draft context that has a populated entity_info, then inject mirrored entities
    # into the draft registry so the selector expansion can see them.
    draft = DraftContext(entity_info=SurfaceMeshEntityInfo(boundaries=[surface]))
    draft._entity_registry.register(mirrored_surface)  # pylint: disable=protected-access

    previewed_names = draft.preview_selector(
        SurfaceSelector(name="mirrored").match("*mirrored"), return_names=True
    )
    assert previewed_names == ["wing_mirrored"]


def test_selector_expansion_with_all_invalid_types_raises_error():
    """
    Test that expansion raises error when ALL matched entities are invalid types.

    This ensures we catch configuration errors where selectors match nothing valid.
    """
    # Create registry with only invalid types for EntityList[Surface]
    registry = EntityRegistry()
    ghost_sphere = GhostSphere(name="ghost_sphere")
    volume = GenericVolume(name="fluid")  # Not a surface type at all
    mirrored_surface = MirroredSurface(
        name="Some random mirrored surface", surface_id="surface1", mirror_plane_id="plane1"
    )  # Not a surface type at all

    registry.register(ghost_sphere)
    registry.register(volume)
    registry.register(mirrored_surface)

    # Create EntityList[Surface] with selector matching all
    # Selector will match mirrored_surface via expansion map
    # but EntityList[Surface] only accepts Surface
    selector = SurfaceSelector(name="all_surfaces").match("*")

    with pytest.raises(ValueError, match="Can not find any valid entity of type.*Surface"):
        entity_list = EntityList[Surface](
            stored_entities=[],
            selectors=[selector],
        )
        expand_entity_list_selectors_in_place(registry, entity_list)

    print("\n✓ Test passed: Error raised when all matched entities are invalid types")


def test_selector_expansion_with_mixed_explicit_and_selector_entities():
    """
    Test that explicit entities and selector-matched entities both get filtered.

    This verifies that the field validator runs consistently regardless of entity source.
    """
    # Create registry
    registry = EntityRegistry()
    surface1 = Surface(name="wing")
    surface2 = MirroredSurface(
        name="wing_mirrored", surface_id="surface1", mirror_plane_id="plane1"
    )
    ghost_sphere = GhostSphere(name="ghost_sphere")

    registry.register(surface1)
    registry.register(surface2)
    registry.register(ghost_sphere)

    # Create EntityList with explicit Surface and selector matching all
    selector = SurfaceSelector(name="mirrored_surfaces").match("*mirrored")
    entity_list = EntityList[Surface, MirroredSurface](
        stored_entities=[surface1],  # Explicit Surface
        selectors=[selector],  # Will match MirroredSurface
    )

    print("\nBefore expansion:")
    print(f"  Explicit: {[e.name for e in entity_list.stored_entities]}")

    expand_entity_list_selectors_in_place(registry, entity_list)

    print("\nAfter expansion:")
    print(f"  All entities: {[e.name for e in entity_list.stored_entities]}")

    entity_names = {e.name for e in entity_list.stored_entities}

    # Should have both explicit and selector-matched valid entities
    assert "wing" in entity_names, "Explicit Surface should be present"
    assert "wing_mirrored" in entity_names, "Selector-matched MirroredSurface should be present"

    # Should not have invalid types
    assert "ghost_sphere" not in entity_names, "Invalid type should be filtered"

    print("\n✓ Test passed: Mixed explicit and selector entities filtered correctly")
