import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.simulation.draft_context.mirror import (
    MirrorPlane,
    _derive_mirrored_entities_from_actions,
)


def test_mirror_single_call_returns_expected_entities(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        mirror_plane = MirrorPlane(
            name="mirrorX",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        mirrored_body_groups, mirrored_surfaces = draft.mirror(
            entities=[body_group], mirror_plane=mirror_plane
        )

        # 1) Returned mirrored body group token is correct.
        assert len(mirrored_body_groups) == 1
        mirrored_body_group = mirrored_body_groups[0]
        assert mirrored_body_group.geometry_body_group_id == body_group.private_attribute_id
        assert mirrored_body_group.mirror_plane_id == mirror_plane.private_attribute_id
        assert mirrored_body_group.name == f"{body_group.name}_<mirror>"

        # 2) Draft mirror actions store the same mapping.
        assert draft._mirror_status == {
            body_group.private_attribute_id: mirror_plane.private_attribute_id
        }

        # 3) Mirrored surfaces correspond exactly to the surfaces of the mirrored body group.
        entity_info = draft._entity_info
        face_group_to_body_group = entity_info.get_face_group_to_body_group_id_map()
        surfaces_by_name = {surface.name: surface for surface in draft.surfaces.entities}

        expected_surface_ids = {
            surfaces_by_name[surface_name].private_attribute_id
            for surface_name, owning_body_group_id in face_group_to_body_group.items()
            if owning_body_group_id == body_group.private_attribute_id
            and surface_name in surfaces_by_name
        }

        mirrored_surface_ids = {mirrored.surface_id for mirrored in mirrored_surfaces}
        assert mirrored_surface_ids == expected_surface_ids

        for mirrored in mirrored_surfaces:
            original_surface = next(
                surface
                for surface in draft.surfaces.entities
                if surface.private_attribute_id == mirrored.surface_id
            )
            assert mirrored.name == f"{original_surface.name}_<mirror>"
            assert mirrored.mirror_plane_id == mirror_plane.private_attribute_id


def test_mirror_multiple_calls_accumulate_and_derive_from_actions(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        first_plane = MirrorPlane(
            name="mirror1",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )
        second_plane = MirrorPlane(
            name="mirror2",
            normal=(0, 1, 0),
            center=(0, 0, 0) * u.m,
        )

        # First mirror request.
        draft.mirror(entities=[body_group], mirror_plane=first_plane)

        # Second mirror request for the same body group should overwrite the action.
        draft.mirror(entities=[body_group], mirror_plane=second_plane)

        assert draft._mirror_status == {
            body_group.private_attribute_id: second_plane.private_attribute_id
        }

        # Derive the full list of mirrored entities from the accumulated mirror actions.
        all_mirrored_body_groups, all_mirrored_surfaces = _derive_mirrored_entities_from_actions(
            mirror_actions=draft._mirror_status,
            entity_info=draft._entity_info,
            body_groups=draft.body_groups.entities,
            surfaces=draft.surfaces.entities,
            mirror_planes=draft._mirror_planes,
        )

        # Only the latest mirror plane should be reflected.
        assert len(all_mirrored_body_groups) == 1
        mirrored_body_group = all_mirrored_body_groups[0]
        assert mirrored_body_group.geometry_body_group_id == body_group.private_attribute_id
        assert mirrored_body_group.mirror_plane_id == second_plane.private_attribute_id
        assert mirrored_body_group.name == f"{body_group.name}_<mirror>"

        # All mirrored surfaces should also use the latest mirror plane.
        entity_info = draft._entity_info
        face_group_to_body_group = entity_info.get_face_group_to_body_group_id_map()
        surfaces_by_name = {surface.name: surface for surface in draft.surfaces.entities}

        expected_surface_ids = {
            surfaces_by_name[surface_name].private_attribute_id
            for surface_name, owning_body_group_id in face_group_to_body_group.items()
            if owning_body_group_id == body_group.private_attribute_id
            and surface_name in surfaces_by_name
        }

        mirrored_surface_ids = {mirrored.surface_id for mirrored in all_mirrored_surfaces}
        assert mirrored_surface_ids == expected_surface_ids

        for mirrored in all_mirrored_surfaces:
            original_surface = next(
                surface
                for surface in draft.surfaces.entities
                if surface.private_attribute_id == mirrored.surface_id
            )
            assert mirrored.name == f"{original_surface.name}_<mirror>"
            assert mirrored.mirror_plane_id == second_plane.private_attribute_id
