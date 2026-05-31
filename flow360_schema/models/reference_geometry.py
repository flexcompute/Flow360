"""Reference geometry model for simulation parameters."""

from __future__ import annotations

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.expression.value_or_expression import ValueOrExpression
from flow360_schema.framework.physical_dimensions import Area, Length


class ReferenceGeometry(Flow360BaseModel):
    """
    :class:`ReferenceGeometry` class contains all geometrical related reference values.

    Example
    -------
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=(1, 1, 1) * u.m,
    ...     area=1.5 * u.m**2
    ... )
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=1 * u.m,
    ...     area=1.5 * u.m**2
    ... )  # Equivalent to above

    ====
    """

    moment_center: Length.Vector3 | None = pd.Field(None, description="The x, y, z coordinate of moment center.")
    moment_length: Length.PositiveFloat64 | Length.PositiveVector3 | None = pd.Field(
        None, description="The x, y, z component-wise moment reference lengths."
    )
    area: ValueOrExpression[Area.PositiveFloat64] | None = pd.Field(
        None, description="The reference area of the geometry."
    )
    private_attribute_area_settings: dict | None = pd.Field(None)

    @classmethod
    def fill_defaults(cls, ref, params):  # type: ignore[override]
        """Return a new ReferenceGeometry with defaults filled using SimulationParams."""
        base_length_unit = params.base_length

        if ref is None:
            ref = cls()

        area = ref.area
        if area is None:
            area = 1.0 * (base_length_unit**2)

        moment_center = ref.moment_center
        if moment_center is None:
            moment_center = (0, 0, 0) * base_length_unit

        moment_length = ref.moment_length
        if moment_length is None:
            moment_length = (1.0, 1.0, 1.0) * base_length_unit

        return cls(
            area=area,
            moment_center=moment_center,
            moment_length=moment_length,
            private_attribute_area_settings=ref.private_attribute_area_settings,
        )
