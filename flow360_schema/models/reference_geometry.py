"""Reference geometry model for simulation parameters."""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import pydantic as pd
from pydantic import Discriminator, Tag

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.expression.value_or_expression import ValueOrExpression
from flow360_schema.framework.physical_dimensions import Area, Length
from flow360_schema.models.entities.surface_entities import Surface


# `computed` stays outside the recipe fields so it remains an output-only result.
class ProjectedArea(Flow360BaseModel):
    """Recipe for automatically computing a projected reference area."""

    model_config = pd.ConfigDict(json_schema_mode_override="serialization")

    type_name: Literal["projected_area"] = pd.Field("projected_area", frozen=True)
    surfaces: EntityList[Surface] = pd.Field(
        description="Explicit surfaces and surface selectors used for the projected-area calculation."
    )
    direction: Literal["X", "Y", "Z"] = pd.Field("X", description="Global projection direction.")
    render_quality: Literal["low", "medium", "high", "ultra"] = pd.Field(
        "medium",
        description="Raster resolution used for projected-area calculation, and therefore "
        "the accuracy of the result. The area counts whole pixels of a rasterized "
        "silhouette, so the error is proportional to pixel size: each step up halves it, "
        "at roughly four times the computation.",
    )
    # Positive area preserves the reference-area constraint and uses the standard physical wire format.
    _computed: Area.PositiveFloat64 | None = pd.PrivateAttr(None)

    def __init__(self, /, **data: Any) -> None:
        if "computed" in data:
            raise TypeError("`computed` is output-only and cannot be passed to ProjectedArea(...).")
        super().__init__(**data)

    @pd.model_validator(mode="wrap")
    @classmethod
    def _deserialize_computed(cls, data: Any, handler: pd.ValidatorFunctionWrapHandler) -> Any:
        if not isinstance(data, dict):
            return handler(data)

        data = data.copy()
        computed = data.pop("computed", None)
        result = handler(data)
        if computed is not None:
            result._set_computed(value=computed)
        return result

    @pd.computed_field(
        return_type=Area.PositiveFloat64 | None,
        repr=False,
        json_schema_extra={
            "readOnly": True,
            "description": "Latest projected-area value computed before submission in SI units.",
        },
    )
    @property
    def computed(self) -> Area.PositiveFloat64 | None:
        """Return the latest computed result, if automatic calculation has run."""
        return self._computed

    def _set_computed(self, *, value: Area.PositiveFloat64) -> None:
        """Set the computed result from submission preparation code."""
        self._computed = pd.TypeAdapter(Area.PositiveFloat64).validate_python(value)

    def preprocess(
        self,
        *,
        params: Any = None,
        exclude: list[str] | None = None,
        required_by: list[str] | None = None,
        flow360_unit_system: Any = None,
    ) -> ProjectedArea:
        """Preserve and nondimensionalize the output-only result for translation."""
        result = cast(
            "ProjectedArea",
            super().preprocess(
                params=params,
                exclude=exclude,
                required_by=required_by,
                flow360_unit_system=flow360_unit_system,
            ),
        )
        if self.computed is not None:
            result._set_computed(value=self.computed.in_base(flow360_unit_system))
        return result

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        """Keep the output-only computed property optional in generated schemas."""
        schema = handler(core_schema)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [field for field in required if field != "computed"]
        return schema


def _reference_area_discriminator(value: Any) -> str:
    """Route projected-area recipes separately from values and expressions."""
    if isinstance(value, ProjectedArea):
        return "projected_area"
    if isinstance(value, dict) and (value.get("type_name") or value.get("typeName")) == "projected_area":
        return "projected_area"
    return "value_or_expression"


ReferenceArea = Annotated[
    Annotated[ProjectedArea, Tag("projected_area")]
    | Annotated[ValueOrExpression[Area.PositiveFloat64], Tag("value_or_expression")],
    Discriminator(_reference_area_discriminator),
]


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
    area: ReferenceArea | None = pd.Field(None, description="The reference area of the geometry.")
    private_attribute_area_settings: dict | None = pd.Field(
        None,
        description="Deprecated Web user interface state retained only when reading legacy simulation data.",
    )

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
