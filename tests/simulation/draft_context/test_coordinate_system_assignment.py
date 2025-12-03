import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemManager,
    CoordinateSystemParent,
    CoordinateSystemStatus,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.simulation_params import SimulationParams
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
        matrix = draft.coordinate_systems.get_coordinate_system_matrix(coordinate_system=assigned)
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


def test_to_status_and_from_status_round_trip(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        body_group = list(draft.body_groups)[0]
        cs_root = draft.coordinate_systems.add(coordinate_system=CoordinateSystem(name="root"))
        cs_child = draft.coordinate_systems.add(
            coordinate_system=CoordinateSystem(name="child"), parent=cs_root
        )
        draft.coordinate_systems.assign(entities=body_group, coordinate_system=cs_child)

        status = draft.coordinate_systems._to_status(
            entity_registry=draft._entity_registry  # pylint:disable=protected-access
        )

        restored = CoordinateSystemManager._from_status(status=status)

        restored_child = restored.get_by_name("child")
        assert restored_child.private_attribute_id == cs_child.private_attribute_id
        restored_assignment = restored.get_for_entity(entity=body_group)
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
            Flow360RuntimeError,
            match="Parent record references unknown coordinate system 'missing'",
        ):
            CoordinateSystemManager._from_status(status=status)


def test_from_status_rejects_duplicate_cs_id(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="cs")
        status = CoordinateSystemStatus(
            coordinate_systems=[cs, cs],
            parents=[],
            assignments=[],
        )
        with pytest.raises(
            Flow360RuntimeError,
            match="Duplicate coordinate system id",
        ):
            CoordinateSystemManager._from_status(status=status)


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
            Flow360RuntimeError,
            match="Assignment references unknown coordinate system 'missing'",
        ):
            CoordinateSystemManager._from_status(status=status)


def test_from_status_rejects_duplicate_entity_assignment(mock_geometry):
    with create_draft(new_run_from=mock_geometry) as draft:
        cs = CoordinateSystem(name="cs")
        status = CoordinateSystemStatus(
            coordinate_systems=[cs],
            parents=[],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id=cs.private_attribute_id,
                    entities=[
                        CoordinateSystemEntityRef(entity_type="Surface", entity_id="s1"),
                        CoordinateSystemEntityRef(entity_type="Surface", entity_id="s1"),
                    ],
                )
            ],
        )
        with pytest.raises(
            Flow360RuntimeError,
            match="Duplicate entity assignment for entity 'Surface:s1'",
        ):
            CoordinateSystemManager._from_status(status=status)


def test_coordinate_system_status_round_trip_through_asset_cache(mock_geometry, tmp_path):
    mock_geometry.internal_registry = mock_geometry._entity_info.get_registry(
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
    uploaded_geometry.internal_registry = uploaded_geometry._entity_info.get_registry(
        uploaded_geometry.internal_registry
    )

    with create_draft(new_run_from=uploaded_geometry) as restored:
        restored_target = list(restored.body_groups)[0]
        restored_assignment = restored.coordinate_systems.get_for_entity(entity=restored_target)
        assert restored_assignment is not None
        assert restored_assignment.name == "child"
