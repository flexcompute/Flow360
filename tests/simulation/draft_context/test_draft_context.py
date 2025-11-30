from flow360.component.project import create_draft
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
