import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.exceptions import Flow360RuntimeError


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

        assigned = draft.coordinate_systems.get_for_entity(entity=body_group)
        assert assigned is not None
        assert assigned.private_attribute_id == child.private_attribute_id

        # Ensure composed matrix matches manual composition using parent-child relationship.
        expected = _compose(root.get_transformation_matrix(), child.get_transformation_matrix())
        matrix = draft.coordinate_systems.get_coordinate_system_matrix(assigned)
        assert matrix is not None
        np.testing.assert_allclose(matrix, expected)

        draft.coordinate_systems.clear_assignment(entity=body_group)
        assert draft.coordinate_systems.get_for_entity(entity=body_group) is None


def test_assign_will_register_when_parent_known(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]

        root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))

        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=root
        )
        draft.coordinate_systems.assign(entities=[body_group], coordinate_system=child)

        assert child in draft.coordinate_systems._coordinate_systems
        assert draft.coordinate_systems.get_for_entity(entity=body_group) == child


def test_assign_coordinate_system_rejects_missing_parent(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        with pytest.raises(
            Flow360RuntimeError,
            match="Parent coordinate system must be registered in the draft before being referenced",
        ):
            draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(name="orphan"),
                parent=CoordinateSystem(name="ghost"),
            )


def test_assign_coordinate_system_rejects_cycle(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs_root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))
        cs_child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=cs_root
        )

        with pytest.raises(
            Flow360RuntimeError, match="Cycle detected in coordinate system inheritance"
        ):
            draft.coordinate_systems.update_parent(coordinate_system=cs_root, parent=cs_child)


def test_assign_coordinate_system_rejects_duplicate_ids(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="first", private_attribute_id="dup-id")
        )
        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system id 'dup-id' already registered.",
        ):
            draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(name="second", private_attribute_id="dup-id")
            )


def test_assign_coordinate_system_rejects_duplicate_names(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="dup-name"))
        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system name 'dup-name' already registered.",
        ):
            draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="dup-name"))


def test_get_coordinate_system_by_name(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="by-name"))
        fetched = draft.coordinate_systems.get_by_name("by-name")
        assert fetched.private_attribute_id == cs.private_attribute_id

        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system 'missing' not found in the draft.",
        ):
            draft.coordinate_systems.get_by_name("missing")


def test_update_parent_requires_registered_entities(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="standalone")
        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system must be part of the draft to be updated.",
        ):
            draft.coordinate_systems.update_parent(coordinate_system=cs, parent=None)

        registered = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="registered")
        )
        unregistered_parent = CoordinateSystem(name="unregistered-parent")
        with pytest.raises(
            Flow360RuntimeError,
            match="Parent coordinate system must be registered in the draft before being referenced.",
        ):
            draft.coordinate_systems.update_parent(
                coordinate_system=registered, parent=unregistered_parent
            )


def test_remove_coordinate_system_errors(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="not-registered")
        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system is not registered in this draft.",
        ):
            draft.coordinate_systems.remove(coordinate_system=cs)

        parent = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="parent"))
        child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=parent
        )
        with pytest.raises(
            Flow360RuntimeError,
            match="Cannot remove coordinate system 'parent' because dependents exist: child",
        ):
            draft.coordinate_systems.remove(coordinate_system=parent)

        # Removing child succeeds
        draft.coordinate_systems.remove(coordinate_system=child)
        assert draft.coordinate_systems.get_for_entity(entity=list(draft.body_groups)[0]) is None


def test_assign_requires_registered_entity(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="cs"))
        rogue_entity = CoordinateSystem(name="not-an-entity")  # wrong type
        with pytest.raises(
            Flow360RuntimeError,
            match="Only entities can be assigned a coordinate system. Received: CoordinateSystem",
        ):
            draft.coordinate_systems.assign(entities=rogue_entity, coordinate_system=cs)


def test_get_coordinate_system_matrix_requires_registration(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="unregistered")
        with pytest.raises(
            Flow360RuntimeError,
            match="Coordinate system must be registered to compute its matrix.",
        ):
            draft.coordinate_systems.get_coordinate_system_matrix(coordinate_system=cs)
