import numpy as np

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_operation import (
    CoordinateSystem,
    Transformation,
)


def _compose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_rotation = parent[:, :3]
    parent_translation = parent[:, 3]

    child_rotation = child[:, :3]
    child_translation = child[:, 3]

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation + parent_translation

    return np.hstack([combined_rotation, combined_translation[:, np.newaxis]])


def test_coordinate_system_transformation_matches_legacy_transformation():
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
            translation=(0, 1, 0) * u.m,
            axis_of_rotation=(0, 0, 1),
            angle_of_rotation=90 * u.deg,
        )
        leaf = CoordinateSystem(
            name="leaf",
            translation=(1, 0, 0) * u.m,
        )

        # Manually compose (root ∘ child ∘ leaf)
        expected = _compose(
            root.get_transformation_matrix(),
            _compose(child.get_transformation_matrix(), leaf.get_transformation_matrix()),
        )

        composed = expected

    np.testing.assert_allclose(composed, expected)
