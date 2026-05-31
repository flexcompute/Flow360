"""Operations that can be performed on entities."""

from __future__ import annotations

from typing import Any, Literal

import pydantic as pd
import unyt
from pydantic import PositiveFloat

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.physical_dimensions import Angle, Length

NDArray = Any


def rotation_matrix_from_axis_and_angle(axis: NDArray, angle: float) -> NDArray:
    """Get rotation matrix from axis and angle of rotation using Rodrigues' formula."""
    import numpy as np

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta
    n_x, n_y, n_z = axis

    # Skew-symmetric cross-product matrix of axis
    cross_n = np.array([[0, -n_z, n_y], [n_z, 0, -n_x], [-n_y, n_x, 0]])

    return np.eye(3) + sin_theta * cross_n + one_minus_cos * np.dot(cross_n, cross_n)


def _build_transformation_matrix(
    *,
    origin: Length.Vector3,  # type: ignore[valid-type]
    axis_of_rotation: Axis,
    angle_of_rotation: Angle.Float64,  # type: ignore[valid-type]
    scale: tuple[PositiveFloat, PositiveFloat, PositiveFloat],
    translation: Length.Vector3,  # type: ignore[valid-type]
) -> NDArray:
    """
    Derive a 3(row) x 4(column) transformation matrix and store as row major.
    Applies to vector of [x, y, z, 1] in project length unit.
    """

    import numpy as np

    origin_array = np.asarray(origin.value)  # type: ignore[attr-defined]
    translation_array = np.asarray(translation.value)  # type: ignore[attr-defined]

    axis = np.asarray(axis_of_rotation, dtype=np.float64)
    angle = angle_of_rotation.to("rad").v.item()  # type: ignore[attr-defined]

    axis = axis / np.linalg.norm(axis)

    rotation_scale_matrix = rotation_matrix_from_axis_and_angle(axis, angle) * np.array(scale)
    final_translation = -rotation_scale_matrix @ origin_array + origin_array + translation_array

    return np.hstack([rotation_scale_matrix, final_translation[:, np.newaxis]])


def _resolve_transformation_matrix(
    *,
    origin: Length.Vector3,  # type: ignore[valid-type]
    axis_of_rotation: Axis,
    angle_of_rotation: Angle.Float64,  # type: ignore[valid-type]
    scale: tuple[PositiveFloat, PositiveFloat, PositiveFloat],
    translation: Length.Vector3,  # type: ignore[valid-type]
    private_attribute_matrix: list[float] | None = None,
) -> NDArray:
    """
    Return the local transformation matrix, honoring a precomputed matrix if provided.
    """
    import numpy as np

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


def _compose_transformation_matrices(parent: NDArray, child: NDArray) -> NDArray:
    """
    Compose two 3x4 transformation matrices (parent . child).
    """
    import numpy as np

    parent_rotation = parent[:, :3]
    parent_translation = parent[:, 3]

    child_rotation = child[:, :3]
    child_translation = child[:, 3]

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation + parent_translation

    return np.hstack([combined_rotation, combined_translation[:, np.newaxis]])


def _transform_point(point: NDArray, matrix: NDArray) -> NDArray:
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


def _transform_direction(vector: NDArray, matrix: NDArray) -> NDArray:
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


def _extract_scale_from_matrix(matrix: NDArray) -> NDArray:
    """
    Extract scale factors from a 3x4 transformation matrix.

    Args:
        matrix: 3x4 transformation matrix

    Returns:
        Scale factors as numpy array [sx, sy, sz]
    """
    import numpy as np

    rotation_scale = matrix[:, :3]
    # Scale factors are the norms of the column vectors
    return np.linalg.norm(rotation_scale, axis=0)


def _is_uniform_scale(matrix: NDArray, rtol: float = 1e-5) -> bool:
    """
    Check if a transformation matrix represents uniform scaling.

    Args:
        matrix: 3x4 transformation matrix
        rtol: Relative tolerance for comparison

    Returns:
        True if the matrix has uniform scaling (sx = sy = sz), False otherwise
    """
    import numpy as np

    scale_factors = _extract_scale_from_matrix(matrix)
    return bool(np.allclose(scale_factors, scale_factors[0], rtol=rtol))


def _validate_uniform_scale_and_transform_center(matrix: NDArray, center: Any, entity_name: str) -> tuple[Any, Any]:
    """
    Common transformation logic for volume primitives that require uniform scaling.

    Validates that the transformation matrix has uniform scaling, extracts the scale factor,
    and transforms the center point.

    Args:
        matrix: 3x4 transformation matrix
        center: The center point (LengthType.Point) to transform
        entity_name: Name of the entity type (e.g., "Sphere", "Cylinder") for error messages

    Returns:
        Tuple of (new_center, uniform_scale) where:
        - new_center: Transformed center point with same type and units as input
        - uniform_scale: The uniform scale factor extracted from the matrix

    Raises:
        Flow360ValueError: If the matrix has non-uniform scaling
    """
    import numpy as np

    if not _is_uniform_scale(matrix):
        scale_factors = _extract_scale_from_matrix(matrix)
        raise Flow360ValueError(
            f"{entity_name} only supports uniform scaling. " f"Detected scale factors: {scale_factors}"
        )

    uniform_scale = _extract_scale_from_matrix(matrix)[0]

    # Transform center
    center_array = np.asarray(center.value)
    new_center_array = _transform_point(center_array, matrix)
    new_center = type(center)(new_center_array, center.units)

    return new_center, uniform_scale


def _extract_rotation_matrix(matrix: NDArray) -> NDArray:
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
    return rotation_scale / scale_factors


def _rotation_matrix_to_axis_angle(rotation_matrix: NDArray) -> tuple[NDArray, float]:
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
    import numpy as np

    # Identity matrix (no rotation)
    if np.allclose(rotation_matrix, np.eye(3)):
        return np.array([1.0, 0.0, 0.0]), 0.0

    # Compute the rotation angle from the trace
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

    # 180-degree rotation (special case)
    if np.abs(angle - np.pi) < 1e-10:
        # For 180-degree rotation, the axis is the eigenvector with eigenvalue 1
        diag = np.diag(rotation_matrix)
        k = np.argmax(diag)

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
    """

    type_name: Literal["CoordinateSystem"] = pd.Field("CoordinateSystem", frozen=True)

    name: str = pd.Field(description="Name of the coordinate system.")
    reference_point: Length.Vector3 = pd.Field(  # type: ignore[valid-type]
        (0, 0, 0) * unyt.m,
        description="Reference point about which scaling and rotation are performed. "
        "Translation is applied after scale and rotation.",
    )

    axis_of_rotation: Axis = pd.Field((1, 0, 0))  # type: ignore[assignment]
    angle_of_rotation: Angle.Float64 = pd.Field(0 * unyt.degree)  # type: ignore[valid-type]

    scale: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = pd.Field((1, 1, 1))

    translation: Length.Vector3 = pd.Field((0, 0, 0) * unyt.m)  # type: ignore[valid-type]
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _get_local_matrix(self) -> NDArray:
        """Local transformation without applying inheritance."""
        return _resolve_transformation_matrix(
            origin=self.reference_point,
            axis_of_rotation=self.axis_of_rotation,
            angle_of_rotation=self.angle_of_rotation,
            scale=self.scale,
            translation=self.translation,
        )

    def get_transformation_matrix(self) -> NDArray:
        """
        Find 3(row)x4(column) transformation matrix and store as row major.
        Applies to vector of [x, y, z, 1] in project length unit.
        """
        return self._get_local_matrix()
