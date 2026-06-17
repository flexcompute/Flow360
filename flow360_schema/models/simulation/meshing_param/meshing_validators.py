"""Shared validation helpers for meshing parameters."""

from typing import Any

import unyt as u

from flow360_schema.models.entities.volume_entities import Box, Cylinder


def validate_snappy_uniform_refinement_entities(refinement: Any) -> None:
    """Validate that a UniformRefinement's entities are compatible with snappyHexMesh.

    Raises ValueError if any Box has a non-axis-aligned rotation or any Cylinder is hollow.
    """
    for entity in refinement.entities.stored_entities:
        if isinstance(entity, Box) and entity.angle_of_rotation.to("deg") % (360 * u.deg) != 0 * u.deg:
            raise ValueError(
                "UniformRefinement for snappy accepts only Boxes with axes aligned"
                + " with the global coordinate system (angle_of_rotation=0)."
            )
        if isinstance(entity, Cylinder) and entity.inner_radius is not None and entity.inner_radius.to("m") != 0 * u.m:
            raise ValueError("UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0).")


def validate_geometry_ai_size_ordering(
    *,
    geometry_accuracy: Any,
    sealing_size: Any,
    min_passage_size: Any,
    location: str,
) -> None:
    """Enforce GeometryAI size ordering, raising ValueError on the first violation.

    - ``min_passage_size`` must not be smaller than ``geometry_accuracy``.
    - When ``sealing_size`` > 0, ``min_passage_size`` must not be smaller than ``sealing_size``.
    - When ``sealing_size`` > 0, ``sealing_size`` must not be smaller than ``geometry_accuracy``.

    ``None`` values are skipped (the constraint they appear in is not checked).
    """
    has_sealing = sealing_size is not None and sealing_size > 0 * u.m

    if has_sealing and geometry_accuracy is not None and sealing_size < geometry_accuracy:
        raise ValueError(
            f"sealing_size ({sealing_size}) must not be smaller than "
            f"geometry_accuracy ({geometry_accuracy}) {location}."
        )

    if min_passage_size is not None:
        if geometry_accuracy is not None and min_passage_size < geometry_accuracy:
            raise ValueError(
                f"min_passage_size ({min_passage_size}) must not be smaller than "
                f"geometry_accuracy ({geometry_accuracy}) {location}."
            )
        if has_sealing and min_passage_size < sealing_size:
            raise ValueError(
                f"min_passage_size ({min_passage_size}) must not be smaller than "
                f"sealing_size ({sealing_size}) {location}."
            )
