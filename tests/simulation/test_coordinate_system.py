import numpy as np

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_operation import (
    CoordinateSystem,
    Transformation,
)


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
