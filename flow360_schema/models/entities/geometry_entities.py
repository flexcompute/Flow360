"""Geometry entity definitions: Edge, GeometryBodyGroup, MirrorPlane, SnappyBody."""
# mypy: disable-error-code="import-not-found"

from __future__ import annotations

from typing import Any, Literal, NoReturn, final

import pydantic as pd

from flow360_schema.exceptions import Flow360DeprecationError
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_operation import (
    _transform_direction,
    _transform_point,
)
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.physical_dimensions import Length

NDArray = Any


@final
class Edge(EntityBase):
    """Edge which contains a set of grouped edges from geometry."""

    private_attribute_entity_type_name: Literal["Edge"] = pd.Field("Edge", frozen=True)
    private_attribute_tag_key: str | None = pd.Field(
        None,
        description="The tag/attribute string used to group geometry edges to form this `Edge`.",
    )
    private_attribute_sub_components: list[str] | None = pd.Field(
        [], description="The edge ids in geometry that composed into this `Edge`."
    )


class GeometryBodyGroup(EntityBase):
    """
    Represents a collection of bodies that are grouped for meshing and
    coordinate-system-based transformation.
    """

    private_attribute_tag_key: str = pd.Field(
        description="The tag/attribute string used to group bodies.",
    )
    private_attribute_entity_type_name: Literal["GeometryBodyGroup"] = pd.Field("GeometryBodyGroup", frozen=True)
    private_attribute_sub_components: list[str] = pd.Field(
        description="A list of body IDs which constitutes the current body group"
    )
    private_attribute_color: str | None = pd.Field(None, description="Color used for visualization")
    mesh_exterior: bool = pd.Field(
        True,
        description="Option to define whether to mesh exterior or interior of body group in geometry AI."
        "Note that this is a beta feature and the interface might change in future releases.",
    )

    @property
    def transformation(self) -> NoReturn:
        """Deprecated property."""
        raise Flow360DeprecationError(
            "GeometryBodyGroup.transformation is deprecated and has been removed. "
            "Please use CoordinateSystem for transformations instead."
        )

    @transformation.setter
    def transformation(self, value: Any) -> NoReturn:
        """Deprecated property setter."""
        raise Flow360DeprecationError(
            "GeometryBodyGroup.transformation is deprecated and has been removed. "
            "Please use CoordinateSystem for transformations instead."
        )


class MirrorPlane(EntityBase):
    """
    Mirror plane entity used by draft runtime to create mirrored entities.

    A ``MirrorPlane`` is an infinite plane defined by a center point and a normal direction.
    """

    normal: Axis = pd.Field(description="Normal direction of the plane.")
    center: Length.Vector3 = pd.Field(description="Center point of the plane.")  # type: ignore[valid-type]

    private_attribute_entity_type_name: Literal["MirrorPlane"] = pd.Field("MirrorPlane", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _apply_transformation(self, matrix: NDArray) -> MirrorPlane:
        """Apply a 3x4 transformation matrix and return a transformed copy."""
        import numpy as np

        center_array = np.asarray(self.center.value)  # type: ignore[attr-defined]
        new_center_array = _transform_point(center_array, matrix)
        new_center = type(self.center)(new_center_array, self.center.units)  # type: ignore[attr-defined, misc]

        normal_array = np.asarray(self.normal)
        transformed_normal = _transform_direction(normal_array, matrix)
        new_normal = tuple(transformed_normal / np.linalg.norm(transformed_normal))

        return self.model_copy(update={"center": new_center, "normal": new_normal})
