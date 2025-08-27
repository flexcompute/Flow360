"""Operations that can be performed on entities."""

from typing import Literal, Optional, Tuple

import numpy as np
import pydantic as pd
from pydantic import PositiveFloat

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.types import Axis


def rotation_matrix_from_axis_and_angle(axis, angle):
    """get rotation matrix from axis and angle of rotation"""
    # Compute the components of the rotation matrix using Rodrigues' formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta

    n_x, n_y, n_z = axis

    # Compute the skew-symmetric cross-product matrix of axis
    cross_n = np.array([[0, -n_z, n_y], [n_z, 0, -n_x], [-n_y, n_x, 0]])

    # Compute the rotation matrix
    rotation_matrix = np.eye(3) + sin_theta * cross_n + one_minus_cos * np.dot(cross_n, cross_n)

    return rotation_matrix


class Transformation(Flow360BaseModel):
    """Transformation that will be applied to a body group."""

    type_name: Literal["BodyGroupTransformation"] = pd.Field("BodyGroupTransformation", frozen=True)

    origin: LengthType.Point = pd.Field(  # pylint:disable=no-member
        (0, 0, 0) * u.m,  # pylint:disable=no-member
        description="The origin for geometry transformation in the order of scale,"
        " rotation and translation.",
    )

    axis_of_rotation: Axis = pd.Field((1, 0, 0))
    angle_of_rotation: AngleType = pd.Field(0 * u.deg)  # pylint:disable=no-member

    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat] = pd.Field((1, 1, 1))

    translation: LengthType.Point = pd.Field((0, 0, 0) * u.m)  # pylint:disable=no-member

    private_attribute_matrix: Optional[list[float]] = pd.Field(None)

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Find 3(row)x4(column) transformation matrix and store as row major.
        Applies to vector of [x, y, z, 1] in project length unit.
        """
        # pylint:disable=no-member
        error_msg = "[Internal] `{}` is dimensioned. Use get_transformation_matrix() after non-dimensionalization!"
        assert str(self.origin.units) == "flow360_length_unit", error_msg.format("origin")
        assert str(self.translation.units) == "flow360_length_unit", error_msg.format("translation")
        origin_array = np.asarray(self.origin.value)
        translation_array = np.asarray(self.translation.value)

        axis = np.asarray(self.axis_of_rotation, dtype=np.float64)
        angle = self.angle_of_rotation.to("rad").v.item()

        axis = axis / np.linalg.norm(axis)

        rotation_scale_matrix = rotation_matrix_from_axis_and_angle(axis, angle) * np.array(
            self.scale
        )
        final_translation = -rotation_scale_matrix @ origin_array + origin_array + translation_array

        return np.hstack([rotation_scale_matrix, final_translation[:, np.newaxis]])
