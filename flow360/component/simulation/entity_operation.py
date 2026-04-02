"""Re-import relay: entity operations now live in flow360-schema."""

# pylint: disable=unused-import
from flow360_schema.framework.entity.entity_operation import (  # noqa: F401
    CoordinateSystem,
    _build_transformation_matrix,
    _compose_transformation_matrices,
    _extract_rotation_matrix,
    _extract_scale_from_matrix,
    _is_uniform_scale,
    _resolve_transformation_matrix,
    _rotation_matrix_to_axis_angle,
    _transform_direction,
    _transform_point,
    _validate_uniform_scale_and_transform_center,
    rotation_matrix_from_axis_and_angle,
)
from flow360_schema.framework.entity.legacy_transformation import (  # noqa: F401
    Transformation,
)
