from flow360.component.simulation.draft_context import DraftContext
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.entity_selector import SurfaceSelector
from flow360.component.simulation.primitives import MirroredSurface, Surface


def test_preview_selector_matches_mirrored_entities():
    """DraftContext.preview_selector should surface mirrored entities from the draft registry."""
    surface = Surface(name="wing")
    mirrored_surface = MirroredSurface(
        name="wing_mirrored", surface_id="surface1", mirror_plane_id="plane1"
    )

    # Build a draft context that has a populated entity_info, then inject mirrored entities
    # into the draft registry so the selector expansion can see them.
    draft = DraftContext(entity_info=SurfaceMeshEntityInfo(boundaries=[surface]))
    draft._entity_registry.register(mirrored_surface)  # pylint: disable=protected-access

    previewed_names = draft.preview_selector(
        SurfaceSelector(name="mirrored").match("*mirrored"), return_names=True
    )
    assert previewed_names == ["wing_mirrored"]
