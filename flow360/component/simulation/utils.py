"""Utility functions for the simulation component."""

from contextlib import contextmanager
from typing import Annotated, Union, get_args, get_origin

import pydantic as pd
from pydantic_core import core_schema


@contextmanager
def model_attribute_unlock(model, attr: str):
    """
    Helper function to set frozen fields of a pydantic model from internal systems
    """
    try:
        # validate_assignment is set to False to allow for the attribute to be modified
        # Otherwise, the attribute will STILL be frozen and cannot be modified
        model.model_config["validate_assignment"] = False
        model.__class__.model_fields[attr].frozen = False
        yield
    finally:
        model.model_config["validate_assignment"] = True
        model.__class__.model_fields[attr].frozen = True


def get_combined_subclasses(cls):
    """get subclasses of cls"""
    if isinstance(cls, tuple):
        subclasses = set()
        for single_cls in cls:
            subclasses.update(single_cls.__subclasses__())
        return list(subclasses)
    return cls.__subclasses__()


def is_exact_instance(obj, cls):
    """Check if an object is an instance of a class and not a subclass."""
    if isinstance(cls, tuple):
        return any(is_exact_instance(obj, c) for c in cls)
    if not isinstance(obj, cls):
        return False
    # Check if there are any subclasses of cls
    subclasses = get_combined_subclasses(cls)
    for subclass in subclasses:
        if isinstance(obj, subclass):
            return False
    return True


def is_instance_of_type_in_union(obj, typ) -> bool:
    """Check whether input `obj` is instance of the types specified in the `Union`(`typ`)"""
    # If typ is an Annotated type, extract the underlying type.
    if get_origin(typ) is Annotated:
        typ = get_args(typ)[0]

    # If the underlying type is a Union, extract its arguments (which are types).
    if get_origin(typ) is Union:
        types_tuple = get_args(typ)
        return isinstance(obj, types_tuple)

    # Otherwise, do a normal isinstance check.
    return isinstance(obj, typ)


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
        """Return the default bounding box with infinite values."""
        return BoundingBox(
            [
                [float("inf"), float("inf"), float("inf")],
                [float("-inf"), float("-inf"), float("-inf")],
            ]
        )

    # --- Pydantic v2 schema integration ---
    @classmethod
    # pylint: disable=unused-argument
    def __get_pydantic_core_schema__(cls, source, handler: pd.GetCoreSchemaHandler):
        # Inner row = 3 floats
        inner_row = core_schema.list_schema(
            core_schema.float_schema(),
            min_length=3,
            max_length=3,
        )
        # Outer list = 2 rows
        outer = core_schema.list_schema(inner_row, min_length=2, max_length=2)
        return core_schema.no_info_after_validator_function(cls._coerce, outer)

    @classmethod
    def _coerce(cls, v):
        # Convert input list into BoundingBox
        return v if isinstance(v, cls) else cls(v)

    # --- Additional geometry helpers ---
    @property
    def size(self):
        """Return the size (dx, dy, dz)."""
        return self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin

    @property
    def center(self):
        """Return the center point of the bounding box."""
        return (
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
            (self.zmin + self.zmax) / 2.0,
        )

    @property
    def largest_dimension(self):
        """Return the largest dimension of the bounding box."""
        return max(self.size)

    def expand(self, other: "BoundingBox") -> "BoundingBox":
        """Return a new bounding box expanded by a given bounding box."""
        (sx0, sy0, sz0), (sx1, sy1, sz1) = self
        (ox0, oy0, oz0), (ox1, oy1, oz1) = other

        # Disabled since if implementation is much faster than using max builtin
        # pylint: disable=consider-using-max-builtin, consider-using-min-builtin
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
