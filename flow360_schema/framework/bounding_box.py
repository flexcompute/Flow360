"""Bounding box types for geometric operations."""

from typing import Annotated, Any

import pydantic as pd
from pydantic_core import CoreSchema, core_schema


class BoundingBox(list[list[float]]):
    """Bounding box."""

    # --- Properties for min/max coordinates ---
    @property
    def xmin(self) -> float:
        """Return the minimum x coordinate."""
        return self[0][0]

    @property
    def ymin(self) -> float:
        """Return the minimum y coordinate."""
        return self[0][1]

    @property
    def zmin(self) -> float:
        """Return the minimum z coordinate."""
        return self[0][2]

    @property
    def xmax(self) -> float:
        """Return the maximum x coordinate."""
        return self[1][0]

    @property
    def ymax(self) -> float:
        """Return the maximum y coordinate."""
        return self[1][1]

    @property
    def zmax(self) -> float:
        """Return the maximum z coordinate."""
        return self[1][2]

    @classmethod
    def get_default_bounding_box(cls) -> "BoundingBox":
        """Return an empty bounding box sentinel initialized for subsequent `expand()` calls.

        The min corner starts at `+inf` and the max corner starts at `-inf`, so this
        default box must be expanded with at least one real bounding box before its
        geometric properties are meaningful.
        """
        return BoundingBox(
            [
                [float("inf"), float("inf"), float("inf")],
                [float("-inf"), float("-inf"), float("-inf")],
            ]
        )

    # --- Pydantic v2 schema integration ---
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: pd.GetCoreSchemaHandler) -> CoreSchema:
        # Inner row = 3 floats
        inner_row = core_schema.list_schema(
            core_schema.float_schema(),
            min_length=3,
            max_length=3,
        )
        # Outer list = 2 rows
        outer = core_schema.list_schema(inner_row, min_length=2, max_length=2)
        return core_schema.no_info_wrap_validator_function(cls._coerce_after_validation, outer)

    @classmethod
    def _coerce(cls, v: Any) -> "BoundingBox":
        # Convert input list into BoundingBox
        return v if isinstance(v, cls) else cls(v)

    @classmethod
    def _coerce_after_validation(cls, v: Any, validator: core_schema.ValidatorFunctionWrapHandler) -> "BoundingBox":
        """Run inner validation first, then coerce the validated list into BoundingBox."""
        return cls._coerce(validator(v))

    # --- Additional geometry helpers ---
    @property
    def size(self) -> tuple[float, float, float]:
        """Return the size (dx, dy, dz)."""
        return self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center point of the bounding box."""
        return (
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
            (self.zmin + self.zmax) / 2.0,
        )

    @property
    def largest_dimension(self) -> float:
        """Return the largest dimension of the bounding box."""
        return max(self.size)

    @property
    def diagonal(self) -> float:
        """Return the diagonal length of the bounding box."""
        dx, dy, dz = self.size
        return (dx**2 + dy**2 + dz**2) ** 0.5  # type: ignore[no-any-return]

    def expand(self, other: "BoundingBox") -> "BoundingBox":
        """Expand this bounding box in-place to include `other` and return self."""
        (sx0, sy0, sz0), (sx1, sy1, sz1) = self
        (ox0, oy0, oz0), (ox1, oy1, oz1) = other

        # Disabled since if implementation is much faster than using max builtin
        if ox0 < sx0:
            sx0 = ox0
        if oy0 < sy0:
            sy0 = oy0
        if oz0 < sz0:
            sz0 = oz0
        if ox1 > sx1:
            sx1 = ox1
        if oy1 > sy1:
            sy1 = oy1
        if oz1 > sz1:
            sz1 = oz1

        self[0][0], self[0][1], self[0][2] = sx0, sy0, sz0
        self[1][0], self[1][1], self[1][2] = sx1, sy1, sz1
        return self


# Annotated alias for documentation
BoundingBoxType = Annotated[
    BoundingBox,
    pd.Field(description="[[xmin, ymin, zmin], [xmax, ymax, zmax]]"),
]
