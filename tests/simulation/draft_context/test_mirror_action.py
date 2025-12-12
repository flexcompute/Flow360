import json
import os

import pytest
from pydantic import ValidationError

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.draft_context.context import DraftContext
from flow360.component.simulation.draft_context.mirror import (
    MirrorPlane,
    MirrorStatus,
    _derive_mirrored_entities_from_actions,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.exceptions import Flow360RuntimeError


def test_mirror_single_call_returns_expected_entities(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        mirror_plane = MirrorPlane(
            name="mirrorX",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        mirrored_body_groups, mirrored_surfaces = draft.mirror.create_mirror_of(
            entities=[body_group], mirror_plane=mirror_plane
        )

        # 1) Returned mirrored body group token is correct.
        assert len(mirrored_body_groups) == 1
        mirrored_body_group = mirrored_body_groups[0]
        assert mirrored_body_group.geometry_body_group_id == body_group.private_attribute_id
        assert mirrored_body_group.mirror_plane_id == mirror_plane.private_attribute_id
        assert mirrored_body_group.name == f"{body_group.name}_<mirror>"

        # 2) Draft mirror actions store the same mapping.
        assert draft.mirror._body_group_id_to_mirror_id == {
            body_group.private_attribute_id: mirror_plane.private_attribute_id
        }

        # 3) Mirrored surfaces correspond exactly to the surfaces of the mirrored body group.
        entity_info = draft._entity_info
        face_group_to_body_group = entity_info.get_face_group_to_body_group_id_map()
        surfaces_by_name = {surface.name: surface for surface in draft.surfaces._entities}

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
                for surface in draft.surfaces._entities
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
        draft.mirror.create_mirror_of(entities=[body_group], mirror_plane=first_plane)

        # Second mirror request for the same body group should overwrite the action.
        draft.mirror.create_mirror_of(entities=[body_group], mirror_plane=second_plane)

        assert draft.mirror._body_group_id_to_mirror_id == {
            body_group.private_attribute_id: second_plane.private_attribute_id
        }

        # Derive the full list of mirrored entities from the accumulated mirror actions.
        # Get face_group_to_body_group map for the test.
        face_group_to_body_group = draft._entity_info.get_face_group_to_body_group_id_map()
        all_mirrored_body_groups, all_mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=draft.mirror._body_group_id_to_mirror_id,
            face_group_to_body_group=face_group_to_body_group,
            body_groups=draft.body_groups._entities,
            surfaces=draft.surfaces._entities,
            mirror_planes=draft.mirror._mirror_planes,
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
        surfaces_by_name = {surface.name: surface for surface in draft.surfaces._entities}

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
                for surface in draft.surfaces._entities
                if surface.private_attribute_id == mirrored.surface_id
            )
            assert mirrored.name == f"{original_surface.name}_<mirror>"
            assert mirrored.mirror_plane_id == second_plane.private_attribute_id


def test_mirror_status_round_trip_through_asset_cache(mock_geometry, tmp_path):
    # Ensure the geometry has an internal registry so that SimulationParams can
    # reference entities and pre-upload processing can update persistent entities.
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
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
        draft.mirror.create_mirror_of(entities=first_body_group, mirror_plane=mirror_plane)
        draft.mirror.create_mirror_of(entities=[second_body_group], mirror_plane=mirror_plane)

        expected_plane_ids = {mirror_plane.private_attribute_id}
        expected_body_group_ids = {
            first_body_group.private_attribute_id,
            second_body_group.private_attribute_id,
        }

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

        mirror_status = processed_params.private_attribute_asset_cache.mirror_status

        # Mirror status must be populated and reference the expected plane/body groups.
        assert isinstance(mirror_status, MirrorStatus)
        assert mirror_status.mirror_planes
        assert mirror_status.mirrored_geometry_body_groups

        plane_ids_in_status = {plane.private_attribute_id for plane in mirror_status.mirror_planes}
        assert plane_ids_in_status == expected_plane_ids

        body_group_ids_in_status = {
            group.geometry_body_group_id for group in mirror_status.mirrored_geometry_body_groups
        }
        assert body_group_ids_in_status == expected_body_group_ids

    # 2. Mimic cloud upload by constructing a geometry asset from the processed params
    #    and ensuring create_draft restores the mirror actions from storage.
    serialized_params = processed_params.model_dump(mode="json")

    with open(os.path.join(tmp_path, "simulation.json"), "w") as f:
        json.dump(serialized_params, f)
    uploaded_geometry = Geometry._from_local_storage(
        asset_id="geo-aaa-aaaa-aaaaaaaa",
        local_storage_path=tmp_path,
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-aaa-aaaa-aaaaaaaa",
                name="Geometry",
                cloud_path_prefix="--",
                status="processed",
            )
        ),
    )
    uploaded_geometry.internal_registry = (
        uploaded_geometry._entity_info.get_persistent_entity_registry(
            uploaded_geometry.internal_registry
        )
    )

    with create_draft(new_run_from=uploaded_geometry) as restored:
        restored_mapping = restored.mirror._body_group_id_to_mirror_id
        expected_mapping = {
            body_group_id: mirror_plane.private_attribute_id
            for body_group_id in expected_body_group_ids
        }
        assert restored_mapping == expected_mapping

        restored_plane_ids = {
            plane.private_attribute_id for plane in restored.mirror._mirror_planes
        }
        assert restored_plane_ids == expected_plane_ids


def test_mirror_create_rejects_duplicate_plane_name(mock_geometry):
    """Test that creating a mirror with a duplicate plane name raises an error."""
    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups, "Test requires at least one body group."

        mirror_plane1 = MirrorPlane(
            name="mirror",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        # First mirror operation should succeed.
        draft.mirror.create_mirror_of(entities=body_groups[0], mirror_plane=mirror_plane1)

        # Second mirror operation with a different plane but same name should fail.
        # Note: We can use the same body group since we're testing plane name uniqueness.
        mirror_plane2 = MirrorPlane(
            name="mirror",  # Same name as mirror_plane1
            normal=(0, 1, 0),
            center=(0, 0, 0) * u.m,
        )

        with pytest.raises(
            Flow360RuntimeError,
            match="Mirror plane name 'mirror' already exists in the draft",
        ):
            draft.mirror.create_mirror_of(entities=body_groups[0], mirror_plane=mirror_plane2)


def test_mirror_from_status_rejects_duplicate_plane_names(mock_geometry):
    """Test that MirrorStatus rejects duplicate mirror plane names via Pydantic validation."""
    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups, "Test requires at least one body group."

        # Create a status with duplicate mirror plane names.
        plane1 = MirrorPlane(name="duplicate", normal=(1, 0, 0), center=(0, 0, 0) * u.m)
        plane2 = MirrorPlane(name="duplicate", normal=(0, 1, 0), center=(0, 0, 0) * u.m)

        # Build a MirrorStatus with duplicate plane names - should fail during construction.
        from flow360.component.simulation.draft_context.mirror import (
            MirroredGeometryBodyGroup,
        )

        with pytest.raises(ValidationError, match="Duplicate mirror plane name 'duplicate'"):
            MirrorStatus(
                mirror_planes=[plane1, plane2],
                mirrored_geometry_body_groups=[
                    MirroredGeometryBodyGroup(
                        name=f"{body_groups[0].name}_<mirror>",
                        geometry_body_group_id=body_groups[0].private_attribute_id,
                        mirror_plane_id=plane1.private_attribute_id,
                    )
                ],
                mirrored_surfaces=[],
            )


def test_remove_mirror_of_removes_mirror_assignment(mock_geometry):
    """Test that remove_mirror_of successfully removes mirror assignments."""
    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups, "Test requires at least one body group."

        body_group = body_groups[0]
        mirror_plane = MirrorPlane(
            name="mirrorX",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        # Create mirror
        mirrored_body_groups, mirrored_surfaces = draft.mirror.create_mirror_of(
            entities=body_group, mirror_plane=mirror_plane
        )

        # Verify mirror was created
        assert len(mirrored_body_groups) == 1
        assert body_group.private_attribute_id in draft.mirror._body_group_id_to_mirror_id

        # Remove mirror
        draft.mirror.remove_mirror_of(entities=body_group)

        # Verify mirror was removed
        assert body_group.private_attribute_id not in draft.mirror._body_group_id_to_mirror_id

        # Verify removing a non-mirrored entity doesn't raise an error
        draft.mirror.remove_mirror_of(entities=body_group)  # Should not raise


def test_remove_mirror_of_rejects_invalid_input_type(mock_geometry):
    """Test that remove_mirror_of rejects invalid input types."""
    with create_draft(new_run_from=mock_geometry) as draft:
        # Try to pass something that's neither a GeometryBodyGroup nor a list
        with pytest.raises(
            Flow360RuntimeError,
            match="`entities` accepts a single entity or a list of entities",
        ):
            draft.mirror.remove_mirror_of(entities="invalid_string")


def test_remove_mirror_of_rejects_invalid_entity_type(mock_geometry):
    """Test that remove_mirror_of rejects invalid entity types."""
    with create_draft(new_run_from=mock_geometry) as draft:
        # Try to remove mirror from a non-GeometryBodyGroup entity (passed as a list)
        invalid_entity = MirrorPlane(name="invalid", normal=(1, 0, 0), center=(0, 0, 0) * u.m)

        with pytest.raises(
            Flow360RuntimeError,
            match="Only GeometryBodyGroup entities are supported by `remove_mirror_of\\(\\)`",
        ):
            draft.mirror.remove_mirror_of(entities=[invalid_entity])


def test_remove_mirror_of_accepts_list_of_entities(mock_geometry):
    """Test that remove_mirror_of accepts a list of entities."""
    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups, "Test requires at least one body group."

        mirror_plane = MirrorPlane(
            name="mirrorX",
            normal=(1, 0, 0),
            center=(0, 0, 0) * u.m,
        )

        # Use first two body groups if available, otherwise just the first one
        entities_to_mirror = [body_groups[0]]
        if len(body_groups) > 1:
            entities_to_mirror.append(body_groups[1])

        # Create mirrors for the body groups
        for body_group in entities_to_mirror:
            draft.mirror.create_mirror_of(entities=body_group, mirror_plane=mirror_plane)

        # Verify mirrors were created
        for body_group in entities_to_mirror:
            assert body_group.private_attribute_id in draft.mirror._body_group_id_to_mirror_id

        # Remove all mirrors at once using list
        draft.mirror.remove_mirror_of(entities=entities_to_mirror)

        # Verify all mirrors were removed
        for body_group in entities_to_mirror:
            assert body_group.private_attribute_id not in draft.mirror._body_group_id_to_mirror_id
