"""Operations that can be performed on entities."""

from typing import Literal, Optional, Tuple

import numpy as np
import pydantic as pd
from pydantic import PositiveFloat

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_utils import generate_uuid
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


def _build_transformation_matrix(
    *,
    origin: LengthType.Point,
    axis_of_rotation: Axis,
    angle_of_rotation: AngleType,
    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
    translation: LengthType.Point,
) -> np.ndarray:
    """
    Derive a 3(row) x 4(column) transformation matrix and store as row major.
    Applies to vector of [x, y, z, 1] in project length unit.
    """

    # pylint:disable=no-member
    origin_array = np.asarray(origin.value)
    translation_array = np.asarray(translation.value)

    axis = np.asarray(axis_of_rotation, dtype=np.float64)
    angle = angle_of_rotation.to("rad").v.item()

    axis = axis / np.linalg.norm(axis)

    rotation_scale_matrix = rotation_matrix_from_axis_and_angle(axis, angle) * np.array(scale)
    final_translation = -rotation_scale_matrix @ origin_array + origin_array + translation_array

    return np.hstack([rotation_scale_matrix, final_translation[:, np.newaxis]])


def _resolve_transformation_matrix(
    *,
    origin: LengthType.Point,
    axis_of_rotation: Axis,
    angle_of_rotation: AngleType,
    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
    translation: LengthType.Point,
    private_attribute_matrix: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Return the local transformation matrix, honoring a precomputed matrix if provided.
    """
    if private_attribute_matrix is not None:
        matrix = np.asarray(private_attribute_matrix, dtype=np.float64)
        matrix = matrix.reshape(3, 4)
        return matrix
    return _build_transformation_matrix(
        origin=origin,
        axis_of_rotation=axis_of_rotation,
        angle_of_rotation=angle_of_rotation,
        scale=scale,
        translation=translation,
    )


def _compose_transformation_matrices(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """
    Compose two 3x4 transformation matrices (parent ∘ child).
    """
    parent_rotation = parent[:, :3]
    parent_translation = parent[:, 3]

    child_rotation = child[:, :3]
    child_translation = child[:, 3]

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation + parent_translation

    return np.hstack([combined_rotation, combined_translation[:, np.newaxis]])


class Transformation(Flow360BaseModel):
    """[Deprecating] Transformation that will be applied to a body group."""

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

        return _resolve_transformation_matrix(
            origin=self.origin,
            axis_of_rotation=self.axis_of_rotation,
            angle_of_rotation=self.angle_of_rotation,
            scale=self.scale,
            translation=self.translation,
        )


class CoordinateSystem(Flow360BaseModel):
    """Coordinate system using geometric transformation primitives."""

    type_name: Literal["CoordinateSystem"] = pd.Field("CoordinateSystem", frozen=True)

    name: str = pd.Field(description="Name of the coordinate system.")
    origin: LengthType.Point = pd.Field(  # pylint:disable=no-member
        (0, 0, 0) * u.m,  # pylint:disable=no-member
        description="The origin for geometry transformation in the order of scale,"
        " rotation and translation.",
    )

    axis_of_rotation: Axis = pd.Field((1, 0, 0))
    angle_of_rotation: AngleType = pd.Field(0 * u.deg)  # pylint:disable=no-member

    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat] = pd.Field((1, 1, 1))

    translation: LengthType.Point = pd.Field((0, 0, 0) * u.m)  # pylint:disable=no-member

    private_attribute_matrix: Optional[list[float]] = pd.Field(
        None, description="Optional precomputed 3x4 transformation matrix in row-major order."
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _get_local_matrix(self) -> np.ndarray:
        """Local transformation without applying inheritance."""
        return _resolve_transformation_matrix(
            origin=self.origin,
            axis_of_rotation=self.axis_of_rotation,
            angle_of_rotation=self.angle_of_rotation,
            scale=self.scale,
            translation=self.translation,
            private_attribute_matrix=self.private_attribute_matrix,
        )

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Find 3(row)x4(column) transformation matrix and store as row major.
        Applies to vector of [x, y, z, 1] in project length unit.
        """
        return self._get_local_matrix()
