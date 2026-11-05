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


def geometry_ai_size_ordering_violation(
    *,
    geometry_accuracy: Any,
    sealing_size: Any,
    min_passage_size: Any,
    location: str,
) -> tuple[str, str] | None:
    """Return the first GeometryAI size-ordering violation as ``(field, message)``, or ``None``.

    ``field`` is the size the message is about: the one that must not be smaller.

    - ``min_passage_size`` must not be smaller than ``geometry_accuracy``.
    - When ``sealing_size`` > 0, ``min_passage_size`` must not be smaller than ``sealing_size``.
    - When ``sealing_size`` > 0, ``sealing_size`` must not be smaller than ``geometry_accuracy``.

    ``None`` values are skipped (the constraint they appear in is not checked).
    """
    has_sealing = sealing_size is not None and sealing_size > 0 * u.m

    if has_sealing and geometry_accuracy is not None and sealing_size < geometry_accuracy:
        return (
            "sealing_size",
            f"sealing_size ({sealing_size}) must not be smaller than "
            f"geometry_accuracy ({geometry_accuracy}) {location}.",
        )

    if min_passage_size is None:
        return None
    if geometry_accuracy is not None and min_passage_size < geometry_accuracy:
        return (
            "min_passage_size",
            f"min_passage_size ({min_passage_size}) must not be smaller than "
            f"geometry_accuracy ({geometry_accuracy}) {location}.",
        )
    if has_sealing and min_passage_size < sealing_size:
        return (
            "min_passage_size",
            f"min_passage_size ({min_passage_size}) must not be smaller than "
            f"sealing_size ({sealing_size}) {location}.",
        )
    return None


# Minimum supported total baffle shell thickness, in multiples of geometry_accuracy.
MIN_BAFFLE_THICKEN_TOTAL_FACTOR = 4.0

# Extent applied to either side when the user leaves it unset. Pinned to the mesher's default.
DEFAULT_BAFFLE_THICKEN_EXTENT = 2.0


def validate_baffle_thicken_total(outward: float | None, inward: float | None, location: str) -> None:
    """Validate that a thickened baffle's shell meets the minimum supported thickness.

    The two extents sum to the total shell thickness. An unset extent resolves to the default.
    """
    outward = DEFAULT_BAFFLE_THICKEN_EXTENT if outward is None else outward
    inward = DEFAULT_BAFFLE_THICKEN_EXTENT if inward is None else inward
    total = outward + inward
    if total < MIN_BAFFLE_THICKEN_TOTAL_FACTOR:
        raise ValueError(
            f"'baffle_thicken_outward' + 'baffle_thicken_inward' is {total:g}x geometry_accuracy "
            f"{location}, below the minimum supported {MIN_BAFFLE_THICKEN_TOTAL_FACTOR}x. Raise "
            "either value, or lower geometry_accuracy so the same physical thickness becomes a "
            "larger multiple of it."
        )
