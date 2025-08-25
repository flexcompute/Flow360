import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.primitives import rotation_matrix_from_axis_and_angle
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.types import Axis


import numpy as np
import pydantic as pd
from pydantic import PositiveFloat


from typing import Literal, Tuple


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