from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box, Cylinder, Sphere
import flow360.component.simulation.units as u


def validate_snappy_uniform_refinement_entities(refinement: UniformRefinement):
    """Validate that a UniformRefinement's entities are compatible with snappyHexMesh.

    Raises ValueError if any Box has a non-axis-aligned rotation or any Cylinder is hollow.
    """
    for entity in refinement.entities.stored_entities:
        if not isinstance(entity, (Box, Cylinder, Sphere)):
            raise ValueError(
                "UniformRefinement for snappy only supports entities of type Box, Cylinder, or Sphere. "
                f"Got {type(entity).__name__} instead."
            )
        if (
            isinstance(entity, Box)
            and entity.angle_of_rotation.to("deg") % (360 * u.deg) != 0 * u.deg
        ):
            raise ValueError(
                "UniformRefinement for snappy accepts only Boxes with axes aligned"
                + " with the global coordinate system (angle_of_rotation=0)."
            )
        if isinstance(entity, Cylinder) and entity.inner_radius.to("m") != 0 * u.m:
            raise ValueError(
                "UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0)."
            )
