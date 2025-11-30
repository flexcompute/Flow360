import copy

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation.draft_context.context import DraftContext
from flow360.component.simulation.draft_context.mirror import (
    MirrorPlane,
    MirrorStatus,
    _derive_mirrored_entities_from_actions,
)
from flow360.component.simulation.simulation_params import SimulationParams


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
        assert draft._body_group_id_to_mirror_id == {
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

        assert draft._body_group_id_to_mirror_id == {
            body_group.private_attribute_id: second_plane.private_attribute_id
        }

        # Derive the full list of mirrored entities from the accumulated mirror actions.
        all_mirrored_body_groups, all_mirrored_surfaces = _derive_mirrored_entities_from_actions(
            mirror_actions=draft._body_group_id_to_mirror_id,
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


def test_mirror_status_round_trip_through_asset_cache(mock_geometry):
    # Ensure the geometry has an internal registry so that SimulationParams can
    # reference entities and pre-upload processing can update persistent entities.
    mock_geometry.internal_registry = mock_geometry._entity_info.get_registry(
        mock_geometry.internal_registry
    )

    # 1. Start from a fresh draft, apply multiple mirror operations, and serialize
    #    mirror status into the SimulationParams asset cache via set_up_params_for_uploading.
    with DraftContext(entity_info=mock_geometry.entity_info) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups, "Test geometry must provide at least one body group."

        first_body_group = body_groups[0]
        second_body_group = body_groups[1] if len(body_groups) > 1 else body_groups[0]

        mirror_plane = MirrorPlane(
            name="mirrorX",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        # Use more than one mirror() call to exercise accumulation logic.
        draft.mirror(entities=[first_body_group], mirror_plane=mirror_plane)
        draft.mirror(entities=[second_body_group], mirror_plane=mirror_plane)

        # Use a minimal SimulationParams that still exercises the
        # pre-upload path without introducing grouping-tag conflicts.
        with u.SI_unit_system:
            params = SimulationParams()

        processed_params = set_up_params_for_uploading(
            root_asset=mock_geometry,
            length_unit=1 * u.m,
            params=params,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

        mirror_status = processed_params.private_attribute_asset_cache.mirror_action

        # Mirror status must be populated and reference the expected plane/body groups.
        assert isinstance(mirror_status, MirrorStatus)
        assert mirror_status.mirror_planes
        assert mirror_status.mirrored_geometry_body_groups

        plane_ids_in_status = {plane.private_attribute_id for plane in mirror_status.mirror_planes}
        assert mirror_plane.private_attribute_id in plane_ids_in_status

        body_group_ids_in_status = {
            group.geometry_body_group_id for group in mirror_status.mirrored_geometry_body_groups
        }
        assert first_body_group.private_attribute_id in body_group_ids_in_status
        assert second_body_group.private_attribute_id in body_group_ids_in_status

    # 2. Mimic cloud upload by constructing a geometry asset from the processed params
    #    and ensuring create_draft restores the mirror actions from storage.
    serialized_params = processed_params.model_dump(mode="json")
    uploaded_geometry = Geometry._from_supplied_simulation_dict(
        serialized_params,
        Geometry(id=None),
    )
    uploaded_geometry.internal_registry = uploaded_geometry._entity_info.get_registry(
        uploaded_geometry.internal_registry
    )
    uploaded_geometry._local_simulation_json = copy.deepcopy(serialized_params)

    with create_draft(new_run_from=uploaded_geometry) as restored:
        restored_mapping = restored._body_group_id_to_mirror_id
        assert restored_mapping

        restored_body_group_ids = set(restored_mapping.keys())
        assert body_group_ids_in_status.issubset(restored_body_group_ids)

        restored_plane_ids = {plane.private_attribute_id for plane in restored.mirror_planes}
        assert restored_plane_ids == plane_ids_in_status
