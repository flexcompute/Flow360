"""
Volume zones parameters
"""

# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

from abc import ABCMeta
from typing import Optional, Union

import pydantic as pd
from pydantic import StrictStr
from typing_extensions import Literal

from ..types import (
    Axis,
    Coordinate,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Vector,
)
from .params_base import Flow360BaseModel
from .unit_system import AngularVelocityType, LengthType


class ReferenceFrameDynamic(Flow360BaseModel):
    """:class:`ReferenceFrameDynamic` class for setting up dynamic reference frame

    Parameters
    ----------
    center : LengthType.Point
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

    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")
    is_dynamic: bool = pd.Field(True, alias="isDynamic", const=True)


class ReferenceFrameExpression(Flow360BaseModel):
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

    theta_radians: Optional[str] = pd.Field(alias="thetaRadians")
    theta_degrees: Optional[str] = pd.Field(alias="thetaDegrees")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = [
            "theta_radians",
            "theta_degrees",
        ]


class ReferenceFrameOmegaRadians(Flow360BaseModel):
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

    omega_radians: float = pd.Field(alias="omegaRadians")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrameOmegaDegrees(Flow360BaseModel):
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

    omega_degrees: float = pd.Field(alias="omegaDegrees")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaDegrees:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrame(Flow360BaseModel):
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

    omega: AngularVelocityType = pd.Field()
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """

        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        omega_radians = solver_values.pop("omega").value
        return ReferenceFrameOmegaRadians(omega_radians=omega_radians, **solver_values)


class VolumeZoneBase(Flow360BaseModel, metaclass=ABCMeta):
    """Basic Boundary class"""

    model_type: str = pd.Field(alias="modelType")


class InitialConditionHeatTransfer(Flow360BaseModel):
    """InitialConditionHeatTransfer"""

    T_solid: Union[PositiveFloat, StrictStr] = pd.Field(options=["Value", "Expression"])


class HeatTransferVolumeZone(VolumeZoneBase):
    """HeatTransferVolumeZone type"""

    model_type = pd.Field("HeatTransfer", alias="modelType", const=True)
    thermal_conductivity: PositiveFloat = pd.Field(alias="thermalConductivity")
    volumetric_heat_source: Optional[Union[NonNegativeFloat, StrictStr]] = pd.Field(
        alias="volumetricHeatSource", options=["Value", "Expression"]
    )
    heat_capacity: Optional[PositiveFloat] = pd.Field(alias="heatCapacity")
    initial_condition: Optional[InitialConditionHeatTransfer] = pd.Field(alias="initialCondition")


class FluidDynamicsVolumeZone(VolumeZoneBase):
    """FluidDynamicsVolumeZone type"""

    model_type = pd.Field("FluidDynamics", alias="modelType", const=True)
    reference_frame: Optional[
        Union[
            ReferenceFrame,
            ReferenceFrameOmegaRadians,
            ReferenceFrameExpression,
            ReferenceFrameDynamic,
        ]
    ] = pd.Field(alias="referenceFrame")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> FluidDynamicsVolumeZone:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


VolumeZoneType = Union[FluidDynamicsVolumeZone, HeatTransferVolumeZone]
