"""
Primitive type definitions for simulation entities.
"""

from abc import ABCMeta
from typing import Literal, Optional, Tuple, Union, final

import numpy as np
import pydantic as pd
from scipy.linalg import eig

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.multi_constructor_model_base import (
    MultiConstructorBaseModel,
)
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.unit_system import AngleType, AreaType, LengthType
from flow360.component.types import Axis


class ReferenceGeometry(Flow360BaseModel):
    """
    Contains all geometrical related refrence values
    Note:
    - mesh_unit is removed from here and will be a property
    TODO:
    - Support expression for time-dependent axis etc?
    - What about force axis?
    """

    # pylint: disable=no-member
    moment_center: Optional[LengthType.Point] = pd.Field(None)
    moment_length: Optional[Union[LengthType.Positive, LengthType.Moment]] = pd.Field(None)
    area: Optional[AreaType.Positive] = pd.Field(None)


class Transformation(Flow360BaseModel):
    """Used in preprocess()/translator to meshing param for volume meshing interface"""

    axis_of_rotation: Optional[Axis] = pd.Field()
    angle_of_rotation: Optional[float] = pd.Field()


class _VolumeEntityBase(EntityBase, metaclass=ABCMeta):
    """All volumetric entities should inherit from this class."""

    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["VolumetricEntityType"] = "VolumetricEntityType"
    private_attribute_zone_boundary_names: Optional[UniqueItemList[str]] = pd.Field(
        None, frozen=True
    )

    def _is_volume_zone(self) -> bool:
        """This is not a zone if zone boundaries are not defined. For validation usage."""
        return self.private_attribute_zone_boundaries is not None


class _SurfaceEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["SurfaceEntityType"] = "SurfaceEntityType"


class _EdgeEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["EdgeEntityType"] = "EdgeEntityType"


@final
class Edge(_EdgeEntityBase):
    """
    Edge with edge name defined in the geometry file
    """

    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["EdgeEntityType"] = pd.Field(
        "EdgeEntityType", frozen=True
    )
    private_attribute_entity_type_name: Literal["Edge"] = pd.Field("Edge", frozen=True)


@final
class GenericVolume(_VolumeEntityBase):
    """
    Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata.
    By design these GenericVolume entities should only contain basic connectivity/mesh information.
    """

    private_attribute_entity_type_name: Literal["GenericVolume"] = pd.Field(
        "GenericVolume", frozen=True
    )
    axes: Optional[Tuple[Axis, Axis]] = pd.Field(None)  # Porous media support
    axis: Optional[Axis] = pd.Field(None)  # Rotation support
    # pylint: disable=no-member
    center: Optional[LengthType.Point] = pd.Field(None)  # Rotation support


@final
class GenericSurface(_SurfaceEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    private_attribute_entity_type_name: Literal["GenericSurface"] = pd.Field(
        "GenericSurface", frozen=True
    )
    private_attribute_is_interface: Optional[bool] = pd.Field(
        False,  # Mostly are not interfaces
        frozen=True,
        description="""This is required in GenericSurface when generated from volume mesh
        but not required when from surface mesh meta.""",
    )


class BoxCache(Flow360BaseModel):
    axes: Optional[Tuple[Axis, Axis]] = pd.Field(None)
    center: Optional[LengthType.Point] = pd.Field(None)
    size: Optional[LengthType.Point] = pd.Field(None)
    name: Optional[str] = pd.Field(None)


@final
class Box(MultiConstructorBaseModel, _VolumeEntityBase):
    """
    Represents a box in three-dimensional space.

    Attributes:
        center (LengthType.Point): The coordinates of the center of the box.
        size (LengthType.Point): The dimensions of the box (length, width, height).
        axes (Tuple[Axis, Axis]]): The axes of the box.
    """

    private_attribute_entity_type_name: Literal["Box"] = pd.Field("Box", frozen=True)
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field()
    size: LengthType.Point = pd.Field()
    # axes: Tuple[Axis, Axis] = pd.Field()
    axis_of_rotation: Axis = pd.Field()
    angle_of_rotation: AngleType = pd.Field()
    private_attribute_input_cache: BoxCache = pd.Field(BoxCache(), frozen=True)

    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_axes(
        cls, name: str, center: LengthType.Point, size: LengthType.Point, axes: Tuple[Axis, Axis]
    ):
        # validate
        x_axis, y_axis = np.array(axes[0]), np.array(axes[1])
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        if not np.isclose(np.linalg.norm(z_axis), 1):
            raise ValueError("Box axes not orthogonal.")

        R = np.transpose(np.asarray([x_axis, y_axis, z_axis], dtype=float))

        # Calculate the rotation axis n
        eigR = eig(R)
        axis = np.real(eigR[1][:, np.where(np.isclose(eigR[0], 1))])
        print(axis.shape)
        if axis.shape[2] > 1:  # in case of 0 rotation angle
            axis = axis[:, :, 0]
        axis = np.ndarray.flatten(axis)

        angle = np.sum(abs(np.angle(eigR[0]))) / 2

        # Find correct angle
        R_test = cls._rotation_matrix_from_axis_and_angle(axis, angle)
        angle *= -1 if np.isclose(R[0, :] @ R_test[:, 0], 1) else 1

        return cls(
            name=name,
            center=center,
            size=size,
            axis_of_rotation=list(axis),
            angle_of_rotation=angle * u.rad,
        )

    @classmethod
    def _rotation_matrix_from_axis_and_angle(cls, axis, angle):
        # Compute the components of the rotation matrix using Rodrigues' formula
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        n_x, n_y, n_z = axis

        # Compute the skew-symmetric cross-product matrix of axis
        K = np.array([[0, -n_z, n_y], [n_z, 0, -n_x], [-n_y, n_x, 0]])

        # Compute the rotation matrix
        R = np.eye(3) + sin_theta * K + one_minus_cos * np.dot(K, K)

        return R

    @pd.model_validator(mode="after")
    def _convert_axis_and_angle_to_coordinate_axes(cls, obj):
        # Ensure the axis is a numpy array
        if not obj.private_attribute_input_cache.axes:
            axis = np.asarray(obj.axis_of_rotation, dtype=float)
            angle = obj.angle_of_rotation.to("rad").v.item()

            # Normalize the axis vector
            axis = axis / np.linalg.norm(axis)

            R = cls._rotation_matrix_from_axis_and_angle(axis, angle)

            obj.private_attribute_input_cache.axes = [list(R[:, 0]), list(R[:, 1])]

        return obj


@final
class Cylinder(_VolumeEntityBase):
    """
    Represents a cylinder in three-dimensional space.

    Attributes:
        axis (Axis): The axis of the cylinder.
        center (LengthType.Point): The center point of the cylinder.
        height (LengthType.Postive): The height of the cylinder.
        inner_radius (LengthType.Positive): The inner radius of the cylinder.
        outer_radius (LengthType.Positive): The outer radius of the cylinder.
    """

    private_attribute_entity_type_name: Literal["Cylinder"] = pd.Field("Cylinder", frozen=True)
    axis: Axis = pd.Field()
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field()
    height: LengthType.Positive = pd.Field()
    inner_radius: Optional[LengthType.Positive] = pd.Field(None)
    # pylint: disable=fixme
    # TODO validation outer > inner
    outer_radius: LengthType.Positive = pd.Field()


@final
class Surface(_SurfaceEntityBase):
    """
    Represents a boudary surface in three-dimensional space.
    """

    private_attribute_entity_type_name: Literal["Surface"] = pd.Field("Surface", frozen=True)

    # pylint: disable=fixme
    # TODO: Should inherit from `ReferenceGeometry` but we do not support this from solver side.


class SurfacePair(Flow360BaseModel):
    """
    Represents a pair of surfaces.

    Attributes:
        pair (Tuple[Surface, Surface]): A tuple containing two Surface objects representing the pair.
    """

    pair: Tuple[Surface, Surface]

    @pd.field_validator("pair", mode="after")
    @classmethod
    def check_unique(cls, v):
        """Check if pairing with self."""
        if v[0].name == v[1].name:
            raise ValueError("A surface cannot be paired with itself.")
        return v

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input(cls, input_data: Union[dict, list, tuple]):
        if isinstance(input_data, (list, tuple)):
            return {"pair": input_data}
        if isinstance(input_data, dict):
            return {"pair": input_data["pair"]}
        raise ValueError("Invalid input data.")

    def __hash__(self):
        return hash(tuple(sorted([self.pair[0].name, self.pair[1].name])))

    def __eq__(self, other):
        if isinstance(other, SurfacePair):
            return tuple(sorted([self.pair[0].name, self.pair[1].name])) == tuple(
                sorted([other.pair[0].name, other.pair[1].name])
            )
        return False

    def __str__(self):
        return ",".join(sorted([self.pair[0].name, self.pair[1].name]))


VolumeEntityTypes = Union[GenericVolume, Cylinder, Box, str]
