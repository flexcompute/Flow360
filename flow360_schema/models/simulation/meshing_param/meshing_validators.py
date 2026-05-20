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
