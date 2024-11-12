"""
Volume zones parameters
"""

# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

from abc import ABCMeta
from typing import Optional, Union

import pydantic.v1 as pd
from numpy import array, dot
from pydantic.v1 import StrictStr
from typing_extensions import Literal

from flow360.component.constants import NumericalConstants
from flow360.component.types import Axis, List
from flow360.component.utils import process_expressions
from flow360.component.v1.params_base import Flow360BaseModel
from flow360.component.v1.unit_system import (
    AngularVelocityType,
    HeatSourceType,
    InverseAreaType,
    InverseLengthType,
    LengthType,
)


class ReferenceFrameBase(Flow360BaseModel):
    """Base reference frame class"""

    model_type: str = pd.Field(alias="modelType")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")
    parent_volume_name: Optional[str] = pd.Field(alias="parentVolumeName")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        exclude_on_flow360_export = ["model_type"]


class ReferenceFrameDynamic(ReferenceFrameBase):
    """:class:`ReferenceFrameDynamic` class for setting up dynamic reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    Returns
    -------
    :class:`ReferenceFrameDynamic`
        An instance of the component class ReferenceFrameDynamic.

    Example
    -------
    >>> rf = ReferenceFrameDynamic(
            center=(0, 0, 0),
            axis=(0, 0, 1),
        )
    """

    model_type: Literal["Dynamic"] = pd.Field("Dynamic", alias="modelType", const=True)

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameDynamic:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrameExpression(ReferenceFrameBase):
    """:class:`ReferenceFrameExpression` class for setting up reference frame using expression

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    parent_volume_name : str, optional
        Name of the volume zone that the rotating reference frame is contained in, used to compute the acceleration in
        the nested rotating reference frame

    theta_radians : str, optional
        Expression for rotation angle (in radians) as a function of time

    theta_degrees : str, optional
        Expression for rotation angle (in degrees) as a function of time


    Returns
    -------
    :class:`ReferenceFrameExpression`
        An instance of the component class ReferenceFrame.

    Example
    -------
    >>> rf = ReferenceFrameExpression(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            theta_radians="1 * t"
        )
    """

    model_type: Literal["Expression"] = pd.Field("Expression", alias="modelType", const=True)
    theta_radians: Optional[str] = pd.Field(alias="thetaRadians")
    theta_degrees: Optional[str] = pd.Field(alias="thetaDegrees")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(ReferenceFrameBase.Config):
        require_one_of = [
            "theta_radians",
            "theta_degrees",
        ]

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameExpression:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrameOmegaRadians(ReferenceFrameBase):
    """:class:`ReferenceFrameOmegaRadians` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega_radians: float
        Nondimensional rotating speed, radians/nondim-unit-time


    Returns
    -------
    :class:`ReferenceFrameOmegaRadians`
        An instance of the component class ReferenceFrameOmegaRadians.

    """

    model_type: Literal["OmegaRadians"] = pd.Field("OmegaRadians", alias="modelType", const=True)
    omega_radians: float = pd.Field(alias="omegaRadians")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrameOmegaDegrees(ReferenceFrameBase):
    """:class:`ReferenceFrameOmegaDegrees` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega_degrees: AngularVelocityType
        Nondimensional rotating speed, radians/nondim-unit-time


    Returns
    -------
    :class:`ReferenceFrameOmegaDegrees`
        An instance of the component class ReferenceFrameOmegaDegrees.

    """

    model_type: Literal["OmegaDegrees"] = pd.Field("OmegaDegrees", alias="modelType", const=True)
    omega_degrees: float = pd.Field(alias="omegaDegrees")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaDegrees:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrame(ReferenceFrameBase):
    """:class:`ReferenceFrame` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega: AngularVelocityType
        Rotating speed, for example radians / s


    Returns
    -------
    :class:`ReferenceFrame`
        An instance of the component class ReferenceFrame.

    Example
    -------
    >>> rf = ReferenceFrame(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            omega=1 * u.rad / u.s
        )
    """

    model_type: Literal["ReferenceFrame"] = pd.Field(
        "ReferenceFrame", alias="modelType", const=True
    )
    omega: AngularVelocityType = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        omega_radians = solver_values.pop("omega").value
        solver_values.pop("model_type", None)
        return ReferenceFrameOmegaRadians(omega_radians=omega_radians, **solver_values)


class VolumeZoneBase(Flow360BaseModel, metaclass=ABCMeta):
    """Basic Boundary class"""

    model_type: str = pd.Field(alias="modelType")


class InitialConditionHeatTransfer(Flow360BaseModel):
    """InitialConditionHeatTransfer"""

    T: Union[pd.PositiveFloat, StrictStr] = pd.Field(options=["Value", "Expression"])

    _processed_T = pd.validator("T", allow_reuse=True)(process_expressions)


ReferenceFrameType = Union[
    ReferenceFrame,
    ReferenceFrameOmegaRadians,
    ReferenceFrameOmegaDegrees,
    ReferenceFrameExpression,
    ReferenceFrameDynamic,
]


class HeatTransferVolumeZone(VolumeZoneBase):
    """HeatTransferVolumeZone type"""

    model_type: Literal["HeatTransfer"] = pd.Field("HeatTransfer", alias="modelType", const=True)
    thermal_conductivity: pd.PositiveFloat = pd.Field(alias="thermalConductivity")
    volumetric_heat_source: Optional[Union[pd.NonNegativeFloat, StrictStr]] = pd.Field(
        alias="volumetricHeatSource", options=["Value", "Expression"]
    )
    heat_capacity: Optional[pd.PositiveFloat] = pd.Field(alias="heatCapacity")
    initial_condition: Optional[InitialConditionHeatTransfer] = pd.Field(alias="initialCondition")


class FluidDynamicsVolumeZone(VolumeZoneBase):
    """FluidDynamicsVolumeZone type"""

    model_type = pd.Field("FluidDynamics", alias="modelType", const=True)
    reference_frame: Optional[ReferenceFrameType] = pd.Field(
        alias="referenceFrame", discriminator="model_type"
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> FluidDynamicsVolumeZone:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class PorousMediumBase(Flow360BaseModel):
    """PorousMediumBase type"""

    darcy_coefficient: InverseAreaType.Point = pd.Field(alias="DarcyCoefficient")
    forchheimer_coefficient: InverseLengthType.Point = pd.Field(alias="ForchheimerCoefficient")
    volumetric_heat_source: Optional[HeatSourceType] = pd.Field(alias="volumetricHeatSource")
    axes: List[Axis] = pd.Field(min_items=2, max_items=3, default=[[1, 0, 0], [0, 1, 0]])

    # pylint: disable=no-self-argument
    @pd.root_validator
    def validate_axes_orthogonal(cls, values):
        """
        check if axes are orthogonal
        """
        axes = values.get("axes")
        x_axis, y_axis = array(axes[0]), array(axes[1])
        max_dot = dot(x_axis, y_axis)
        if len(axes) == 3:
            z_axis = array(axes[2])
            max_dot = max(max_dot, dot(x_axis, z_axis), dot(y_axis, z_axis))
        if max_dot > NumericalConstants.epsilon:
            raise ValueError("Porous zone axes not orthogonal.")

        return values


class PorousMediumVolumeZone(VolumeZoneBase, PorousMediumBase):
    """PorousMediumVolumeZone type"""

    model_type: Literal["PorousMedium"] = pd.Field("PorousMedium", alias="modelType", const=True)


VolumeZoneType = Union[FluidDynamicsVolumeZone, HeatTransferVolumeZone, PorousMediumVolumeZone]
