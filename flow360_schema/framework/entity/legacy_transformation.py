"""Legacy transformation model kept outside schema-generation import paths."""
# mypy: disable-error-code="import-not-found"

from __future__ import annotations

from typing import Any, Literal

import pydantic as pd
import unyt
from pydantic import PositiveFloat

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_operation import _resolve_transformation_matrix
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.physical_dimensions import Angle, Length

NDArray = Any


class Transformation(Flow360BaseModel):
    """[Deprecating] Transformation that will be applied to a body group."""

    type_name: Literal["BodyGroupTransformation"] = pd.Field("BodyGroupTransformation", frozen=True)

    origin: Length.Vector3 = pd.Field(  # type: ignore[valid-type]
        (0, 0, 0) * unyt.m,
        description="The origin for geometry transformation in the order of scale," " rotation and translation.",
    )

    axis_of_rotation: Axis = pd.Field((1, 0, 0))  # type: ignore[assignment]
    angle_of_rotation: Angle.Float64 = pd.Field(0 * unyt.degree)  # type: ignore[valid-type]

    scale: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = pd.Field((1, 1, 1))

    translation: Length.Vector3 = pd.Field((0, 0, 0) * unyt.m)  # type: ignore[valid-type]

    private_attribute_matrix: list[float] | None = pd.Field(None)

    def get_transformation_matrix(self) -> NDArray:
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
            private_attribute_matrix=self.private_attribute_matrix,
        )
