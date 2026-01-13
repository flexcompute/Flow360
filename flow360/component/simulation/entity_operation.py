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


def _resolve_transformation_matrix(  # pylint:disable=too-many-arguments
    *,
    # pylint: disable=no-member
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


def _transform_point(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transform a 3D point using a 3x4 transformation matrix.

    Args:
        point: 3D point as numpy array [x, y, z]
        matrix: 3x4 transformation matrix

    Returns:
        Transformed point as numpy array [x', y', z']
    """
    rotation_scale = matrix[:, :3]
    translation = matrix[:, 3]
    return rotation_scale @ point + translation


def _transform_direction(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transform a direction vector using only the rotation part of a transformation matrix.
    Does not apply translation (directions are independent of position).

    Args:
        vector: 3D direction vector as numpy array [x, y, z]
        matrix: 3x4 transformation matrix

    Returns:
        Transformed direction vector as numpy array [x', y', z']
    """
    rotation_scale = matrix[:, :3]
    return rotation_scale @ vector


def _extract_scale_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Extract scale factors from a 3x4 transformation matrix.

    Args:
        matrix: 3x4 transformation matrix

    Returns:
        Scale factors as numpy array [sx, sy, sz]
    """
    rotation_scale = matrix[:, :3]
    # Scale factors are the norms of the column vectors
    return np.linalg.norm(rotation_scale, axis=0)


def _is_uniform_scale(matrix: np.ndarray, rtol: float = 1e-5) -> bool:
    """
    Check if a transformation matrix represents uniform scaling.

    Args:
        matrix: 3x4 transformation matrix
        rtol: Relative tolerance for comparison

    Returns:
        True if the matrix has uniform scaling (sx = sy = sz), False otherwise
    """
    scale_factors = _extract_scale_from_matrix(matrix)
    return np.allclose(scale_factors, scale_factors[0], rtol=rtol)


def _extract_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Extract the pure rotation matrix from a 3x4 transformation matrix,
    removing any scaling component.

    Args:
        matrix: 3x4 transformation matrix

    Returns:
        Pure 3x3 rotation matrix (orthonormal)
    """
    rotation_scale = matrix[:, :3]
    scale_factors = _extract_scale_from_matrix(matrix)

    # Divide each column by its scale factor to remove scaling
    rotation_matrix = rotation_scale / scale_factors
    return rotation_matrix


def _rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract axis-angle representation from a 3x3 rotation matrix.
    This is the inverse operation of rotation_matrix_from_axis_and_angle.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Tuple of (axis, angle) where:
            - axis: 3D unit vector as numpy array [x, y, z]
            - angle: rotation angle in radians
    """
    # Check for identity matrix (no rotation)
    if np.allclose(rotation_matrix, np.eye(3)):
        return np.array([1.0, 0.0, 0.0]), 0.0

    # Compute the rotation angle from the trace
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

    # Check for 180-degree rotation (special case)
    if np.abs(angle - np.pi) < 1e-10:
        # For 180-degree rotation, the axis is the eigenvector with eigenvalue 1
        # Find the column with the largest diagonal element
        diag = np.diag(rotation_matrix)
        k = np.argmax(diag)

        # Extract axis from the matrix
        axis = np.zeros(3)
        axis[k] = np.sqrt((rotation_matrix[k, k] + 1) / 2)

        for i in range(3):
            if i != k:
                axis[i] = rotation_matrix[k, i] / (2 * axis[k])

        axis = axis / np.linalg.norm(axis)
        return axis, angle

    # General case: extract axis from skew-symmetric part
    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]
    )

    axis = axis / np.linalg.norm(axis)
    return axis, angle


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
    """
    Coordinate system using geometric transformation primitives.

    The transformation is applied in the following order:

    1. **Scale**: Apply scaling factors (sx, sy, sz) about the reference_point
    2. **Rotate**: Rotate by angle_of_rotation about axis_of_rotation through the reference_point
    3. **Translate**: Apply translation vector to the result

    Mathematically, for a point P, the transformation is:
        P' = R * S * (P - reference_point) + reference_point + translation

    where:
        - S is the scaling matrix with diagonal (sx, sy, sz)
        - R is the rotation matrix derived from axis_of_rotation and angle_of_rotation
        - reference_point is the origin for scale and rotation operations
        - translation is the final displacement vector

    Examples
    --------
    Create a coordinate system that scales by 2x, rotates 90° about Z-axis, then translates:

    >>> import flow360 as fl
    >>> cs = fl.CoordinateSystem(
    ...     name="my_frame",
    ...     reference_point=(0, 0, 0) * fl.u.m,
    ...     axis_of_rotation=(0, 0, 1),
    ...     angle_of_rotation=90 * fl.u.deg,
    ...     scale=(2, 2, 2),
    ...     translation=(1, 0, 0) * fl.u.m
    ... )
    """

    type_name: Literal["CoordinateSystem"] = pd.Field("CoordinateSystem", frozen=True)

    name: str = pd.Field(description="Name of the coordinate system.")
    reference_point: LengthType.Point = pd.Field(  # pylint:disable=no-member
        (0, 0, 0) * u.m,  # pylint:disable=no-member
        description="Reference point about which scaling and rotation are performed. "
        "Translation is applied after scale and rotation.",
    )

    axis_of_rotation: Axis = pd.Field((1, 0, 0))
    angle_of_rotation: AngleType = pd.Field(0 * u.deg)  # pylint:disable=no-member

    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat] = pd.Field((1, 1, 1))

    translation: LengthType.Point = pd.Field((0, 0, 0) * u.m)  # pylint:disable=no-member
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _get_local_matrix(self) -> np.ndarray:
        """Local transformation without applying inheritance."""
        return _resolve_transformation_matrix(
            origin=self.reference_point,
            axis_of_rotation=self.axis_of_rotation,
            angle_of_rotation=self.angle_of_rotation,
            scale=self.scale,
            translation=self.translation,
        )

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Find 3(row)x4(column) transformation matrix and store as row major.
        Applies to vector of [x, y, z, 1] in project length unit.
        """
        return self._get_local_matrix()
