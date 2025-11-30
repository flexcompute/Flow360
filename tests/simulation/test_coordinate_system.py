import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_operation import (
    CoordinateSystem,
    Transformation,
    validate_coordinate_systems,
)
from flow360.exceptions import Flow360RuntimeError


def _compose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_rotation = parent[:, :3]
    parent_translation = parent[:, 3]

    child_rotation = child[:, :3]
    child_translation = child[:, 3]

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation + parent_translation

    return np.hstack([combined_rotation, combined_translation[:, np.newaxis]])


def test_udc_transformation_matrix_matches_transformation():
    with u.SI_unit_system:
        transform_args = {
            "origin": (1.0, 2.0, 3.0) * u.m,
            "axis_of_rotation": (0, 0, 1),
            "angle_of_rotation": 90 * u.deg,
            "scale": (2.0, 3.0, 4.0),
            "translation": (5.0, 6.0, 7.0) * u.m,
        }
        transformation_matrix = Transformation(**transform_args).get_transformation_matrix()
        coordinate_matrix = CoordinateSystem(
            name="vehicle_frame", **transform_args
        ).get_transformation_matrix()

    np.testing.assert_allclose(coordinate_matrix, transformation_matrix)


def test_coordinate_system_inheritance_composes_transformations():
    with u.SI_unit_system:
        root = CoordinateSystem(
            name="root",
            translation=(10, 0, 0) * u.m,
        )
        child = CoordinateSystem(
            name="child",
            parent_id=root.private_attribute_id,
            translation=(0, 1, 0) * u.m,
            axis_of_rotation=(0, 0, 1),
            angle_of_rotation=90 * u.deg,
        )
        leaf = CoordinateSystem(
            name="leaf",
            parent_id=child.private_attribute_id,
            translation=(1, 0, 0) * u.m,
        )

        mapping = {
            root.private_attribute_id: root,
            child.private_attribute_id: child,
            leaf.private_attribute_id: leaf,
        }

        # Manually compose (root ∘ child ∘ leaf)
        expected = _compose(
            root.get_transformation_matrix(),
            _compose(child.get_transformation_matrix(), leaf.get_transformation_matrix()),
        )

        composed = leaf.get_transformation_matrix(mapping)

    np.testing.assert_allclose(composed, expected)


def test_coordinate_system_cycle_detection():
    with u.SI_unit_system:
        a = CoordinateSystem(name="a")
        b = CoordinateSystem(name="b", parent_id=a.private_attribute_id)
        # Introduce a cycle.
        a.parent_id = b.private_attribute_id

        mapping = {
            a.private_attribute_id: a,
            b.private_attribute_id: b,
        }

        with pytest.raises(Flow360RuntimeError):
            b.get_transformation_matrix(mapping)


def test_validate_coordinate_systems_missing_parent():
    with u.SI_unit_system:
        orphan = CoordinateSystem(name="orphan", parent_id="missing-parent")

        with pytest.raises(Flow360RuntimeError):
            validate_coordinate_systems({orphan.private_attribute_id: orphan})
