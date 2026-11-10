import flow360_schema.models.simulation.units as u
import numpy as np
import pytest
from flow360_schema.framework.entity.entity_operation import CoordinateSystem
from flow360_schema.models.entities.geometry_entities import Edge
from flow360_schema.models.entities.surface_entities import ImportedSurface
from flow360_schema.models.simulation.simulation_params import SimulationParams

from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemManager,
    CoordinateSystemParent,
    CoordinateSystemStatus,
)
from flow360.exceptions import Flow360ValueError


def _compose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_rotation = parent[:, :3]
    parent_translation = parent[:, 3]

    child_rotation = child[:, :3]
    child_translation = child[:, 3]

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation + parent_translation

    return np.hstack([combined_rotation, combined_translation[:, np.newaxis]])


def test_register_and_assign_coordinate_system(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="vehicle"))
        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(
                name="component",
                translation=(1, 0, 0) * u.m,
            ),
            parent=root,
        )

        draft.coordinate_systems.assign(entities=body_group, coordinate_system=child)

        assigned = draft.coordinate_systems._get_coordinate_system_for_entity(entity=body_group)
        assert assigned is not None
        assert assigned.private_attribute_id == child.private_attribute_id

        # Ensure composed matrix matches manual composition using parent-child relationship.
        expected = _compose(root.get_transformation_matrix(), child.get_transformation_matrix())
        matrix = draft.coordinate_systems._get_coordinate_system_matrix(coordinate_system=assigned)
        assert matrix is not None
        np.testing.assert_allclose(matrix, expected)

        draft.coordinate_systems.clear_assignment(entity=body_group)
        assert draft.coordinate_systems._get_coordinate_system_for_entity(entity=body_group) is None


def test_assign_will_register_when_parent_known(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))

        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=root
        )
        draft.coordinate_systems.assign(entities=[body_group], coordinate_system=child)

        assert child in draft.coordinate_systems._coordinate_systems
        assert (
            draft.coordinate_systems._get_coordinate_system_for_entity(entity=body_group) == child
        )


def test_add_auto_registers_missing_parent(mock_geometry):
    """Test that add() auto-registers a missing parent as root."""
    with create_draft(new_run_from=mock_geometry) as draft:
        parent = CoordinateSystem(name="ghost")
        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="orphan"),
            parent=parent,
        )
        # Parent should be auto-registered as root
        assert parent in draft.coordinate_systems._coordinate_systems
        assert child in draft.coordinate_systems._coordinate_systems
        # Parent should have no parent (root)
        assert (
            draft.coordinate_systems._coordinate_system_parents[parent.private_attribute_id] is None
        )
        # Child should have parent as its parent
        assert (
            draft.coordinate_systems._coordinate_system_parents[child.private_attribute_id]
            == parent.private_attribute_id
        )


def test_assign_coordinate_system_rejects_cycle(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs_root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))
        cs_child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=cs_root
        )

        with pytest.raises(
            Flow360ValueError, match="Cycle detected in coordinate system inheritance"
        ):
            draft.coordinate_systems.update_parent(coordinate_system=cs_root, parent=cs_child)


def test_assign_coordinate_system_rejects_duplicate_ids(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="first", private_attribute_id="dup-id")
        )
        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system id 'dup-id' already registered.",
        ):
            draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(name="second", private_attribute_id="dup-id")
            )


def test_assign_coordinate_system_rejects_duplicate_names(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="dup-name"))
        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system name 'dup-name' already registered.",
        ):
            draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="dup-name"))


def test_get_coordinate_system_by_name(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="by-name"))
        fetched = draft.coordinate_systems.get_by_name("by-name")
        assert fetched.private_attribute_id == cs.private_attribute_id

        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system 'missing' not found in the draft.",
        ):
            draft.coordinate_systems.get_by_name("missing")


def test_update_parent_requires_registered_coordinate_system(mock_geometry):
    """Test that update_parent requires the coordinate system itself to be registered."""
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="standalone")
        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system must be part of the draft to be updated.",
        ):
            draft.coordinate_systems.update_parent(coordinate_system=cs, parent=None)


def test_update_parent_auto_registers_missing_parent(mock_geometry):
    """Test that update_parent auto-registers an unregistered parent as root."""
    with create_draft(new_run_from=mock_geometry) as draft:
        registered = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="registered")
        )
        unregistered_parent = CoordinateSystem(name="unregistered-parent")

        # Should auto-register the parent
        draft.coordinate_systems.update_parent(
            coordinate_system=registered, parent=unregistered_parent
        )

        # Parent should now be registered as root
        assert unregistered_parent in draft.coordinate_systems._coordinate_systems
        assert (
            draft.coordinate_systems._coordinate_system_parents[
                unregistered_parent.private_attribute_id
            ]
            is None
        )
        # registered should now have unregistered_parent as its parent
        assert (
            draft.coordinate_systems._coordinate_system_parents[registered.private_attribute_id]
            == unregistered_parent.private_attribute_id
        )


def test_remove_coordinate_system_errors(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="not-registered")
        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system is not registered in this draft.",
        ):
            draft.coordinate_systems.remove(coordinate_system=cs)

        parent = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="parent"))
        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=parent
        )
        with pytest.raises(
            Flow360ValueError,
            match="Cannot remove coordinate system 'parent' because dependents exist: child",
        ):
            draft.coordinate_systems.remove(coordinate_system=parent)

        # Removing child succeeds
        draft.coordinate_systems.remove(coordinate_system=child)
        assert (
            draft.coordinate_systems._get_coordinate_system_for_entity(
                entity=list(draft.body_groups)[0]
            )
            is None
        )


def test_assign_requires_registered_entity(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="cs"))
        rogue_entity = CoordinateSystem(name="not-an-entity")  # wrong type
        with pytest.raises(
            Flow360ValueError,
            match="Only entities can be assigned a coordinate system. Received: CoordinateSystem",
        ):
            draft.coordinate_systems.assign(entities=rogue_entity, coordinate_system=cs)


def test_transformable_marker_and_transformation_hook_agree():
    """Carrying the marker and implementing the hook must go together.

    ``assign`` checks the marker while the pipeline calls the hook, so a class with one
    and not the other is either wrongly refused or accepted and then silently ignored.
    A body group is the sole exception: the mesher moves its geometry for it.
    """
    from flow360_schema.framework.entity.entity_base import EntityBase
    from flow360_schema.framework.entity.entity_operation import TransformableEntity

    def all_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from all_subclasses(subclass)

    marked, implementing = set(), set()
    for subclass in all_subclasses(EntityBase):
        entity_type = getattr(
            subclass.model_fields.get("private_attribute_entity_type_name"), "default", None
        )
        if not isinstance(entity_type, str):
            continue
        if issubclass(subclass, TransformableEntity):
            marked.add(entity_type)
        if hasattr(subclass, "_apply_transformation"):
            implementing.add(entity_type)

    # Guards against the subclass walk finding nothing and passing vacuously.
    assert marked

    assert not implementing - marked
    # The two entities a consumer moves rather than the entity moving itself: a body
    # group's matrix goes to the surface mesher, an imported surface's to the solver as
    # `transformationMatrix`. Both are assignable, so both are marked without a hook.
    assert marked - implementing == {"GeometryBodyGroup", "ImportedSurface"}


def test_assign_accepts_imported_surface(mock_geometry):
    """An imported surface is moved by the solver, so the assignment must be accepted.

    `inject_imported_surface_info` reads the composed matrix and emits it as
    `transformationMatrix`, so refusing the assignment would break a supported path even
    though the entity implements no transformation hook of its own.
    """
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="cs", translation=(1, 0, 0) * u.m)
        )
        with u.SI_unit_system:
            surface = ImportedSurface(
                name="imported", file_name="imported.cgns", private_attribute_id="imported-1"
            )

        draft.coordinate_systems.assign(entities=surface, coordinate_system=cs)

        assert draft.coordinate_systems._get_matrix_for_entity(entity=surface) is not None


def test_assign_rejects_non_transformable_entity(mock_geometry):
    """A surface carries no transform, so assigning one must fail at the call.

    Nothing downstream reads a coordinate system off a surface, so accepting the
    assignment would leave the geometry unmoved with no indication why.
    """
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="cs"))
        surface = list(draft.surfaces)[0]
        with pytest.raises(
            Flow360ValueError,
            match="A coordinate system cannot be assigned to .*Surface",
        ):
            draft.coordinate_systems.assign(entities=surface, coordinate_system=cs)


def test_get_coordinate_system_matrix_requires_registration(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="unregistered")
        with pytest.raises(
            Flow360ValueError,
            match="Coordinate system must be registered to compute its matrix.",
        ):
            draft.coordinate_systems._get_coordinate_system_matrix(coordinate_system=cs)


def test_to_status_and_from_status_round_trip(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]
        cs_root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))
        cs_child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=cs_root
        )
        draft.coordinate_systems.assign(entities=body_group, coordinate_system=cs_child)

        status = draft.coordinate_systems._to_status()

        restored = CoordinateSystemManager._from_status(
            status=status,
            entity_registry=draft._entity_registry,  # pylint: disable=protected-access
        )

        restored_child = restored.get_by_name("child")
        assert restored_child.private_attribute_id == cs_child.private_attribute_id
        restored_assignment = restored._get_coordinate_system_for_entity(entity=body_group)
        assert restored_assignment.private_attribute_id == cs_child.private_attribute_id


def test_from_status_validation_errors(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        status = CoordinateSystemStatus(
            coordinate_systems=[CoordinateSystem(name="cs1")],
            parents=[CoordinateSystemParent(coordinate_system_id="missing", parent_id=None)],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id="missing",
                    entities=[CoordinateSystemEntityRef(entity_type="Surface", entity_id="surf-1")],
                )
            ],
        )
        with pytest.raises(
            Flow360ValueError,
            match="Parent record references unknown coordinate system 'missing'",
        ):
            CoordinateSystemManager._from_status(
                status=status,
                entity_registry=draft._entity_registry,  # pylint: disable=protected-access
            )


def test_from_status_rejects_assignment_unknown_cs(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        status = CoordinateSystemStatus(
            coordinate_systems=[],
            parents=[],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id="missing",
                    entities=[CoordinateSystemEntityRef(entity_type="Surface", entity_id="s1")],
                )
            ],
        )
        with pytest.raises(
            Flow360ValueError,
            match="Assignment references unknown coordinate system 'missing'",
        ):
            CoordinateSystemManager._from_status(
                status=status,
                entity_registry=draft._entity_registry,  # pylint: disable=protected-access
            )


def test_from_status_rejects_duplicate_entity_assignment(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="cs")
        body_group = list(draft.body_groups)[0]
        entity_type_name = body_group.private_attribute_entity_type_name
        entity_id = body_group.private_attribute_id
        status = CoordinateSystemStatus(
            coordinate_systems=[cs],
            parents=[],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id=cs.private_attribute_id,
                    entities=[
                        CoordinateSystemEntityRef(
                            entity_type=entity_type_name, entity_id=entity_id
                        ),
                        CoordinateSystemEntityRef(
                            entity_type=entity_type_name, entity_id=entity_id
                        ),
                    ],
                )
            ],
        )
        with pytest.raises(
            Flow360ValueError,
            match=f"Duplicate entity assignment for entity '{entity_type_name}:{entity_id}'",
        ):
            CoordinateSystemManager._from_status(
                status=status,
                entity_registry=draft._entity_registry,  # pylint: disable=protected-access
            )


def test_from_status_filters_assignment_to_unknown_entity(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="cs")
        body_group = list(draft.body_groups)[0]
        entity_type_name = body_group.private_attribute_entity_type_name

        status = CoordinateSystemStatus(
            coordinate_systems=[cs],
            parents=[],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id=cs.private_attribute_id,
                    entities=[
                        CoordinateSystemEntityRef(
                            entity_type=entity_type_name, entity_id="missing-entity-id"
                        )
                    ],
                )
            ],
        )

        restored = CoordinateSystemManager._from_status(
            status=status,
            entity_registry=draft._entity_registry,  # pylint: disable=protected-access
        )
        assert (
            restored._entity_key_to_coordinate_system_id == {}  # pylint: disable=protected-access
        )


def test_coordinate_system_status_round_trip_through_asset_cache(mock_geometry, tmp_path):
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )
    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert body_groups
        target = body_groups[0]

        cs_root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))
        cs_child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=cs_root
        )
        draft.coordinate_systems.assign(entities=target, coordinate_system=cs_child)

        with u.SI_unit_system:
            params = SimulationParams()

        processed = set_up_params_for_uploading(
            root_asset=mock_geometry,
            length_unit=1 * u.m,
            params=params,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

        status = processed.private_attribute_asset_cache.coordinate_system_status
        assert isinstance(status, CoordinateSystemStatus)
        assert status.coordinate_systems
        assert status.parents
        assert status.assignments

    serialized = processed.model_dump(mode="json")
    json_path = tmp_path / "simulation.json"
    json_path.write_text(__import__("json").dumps(serialized))

    from flow360.component.geometry import Geometry, GeometryMeta
    from flow360.component.resource_base import local_metadata_builder

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
        restored_target = list(restored.body_groups)[0]
        restored_assignment = restored.coordinate_systems._get_coordinate_system_for_entity(
            entity=restored_target
        )
        assert restored_assignment is not None
        assert restored_assignment.name == "child"


def test_imported_surface_explicit_id_preserved():
    """An explicitly provided private_attribute_id should not be overwritten."""
    with u.SI_unit_system:
        surface = ImportedSurface(
            name="surface1", file_name="surface1.cgns", private_attribute_id="custom_id"
        )
    assert surface.private_attribute_id == "custom_id"
