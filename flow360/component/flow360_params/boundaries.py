"""
Boundaries parameters
"""

from abc import ABCMeta
from typing import Literal, Optional, Tuple, Union

import pydantic as pd
from pydantic import StrictStr

from flow360.component.flow360_params.unit_system import PressureType

from ..types import Axis, PositiveFloat, PositiveInt, Vector
from .params_base import Flow360BaseModel
from .turbulence_quantities import TurbulenceQuantitiesType
from .unit_system import VelocityType

BoundaryVelocityType = Union[VelocityType.Vector, Tuple[StrictStr, StrictStr, StrictStr]]
BoundaryAxisType = Union[Axis, Tuple[StrictStr, StrictStr, StrictStr]]


# pylint: enable=too-many-arguments, too-many-return-statements, too-many-branches
class Boundary(Flow360BaseModel, metaclass=ABCMeta):
    """Basic Boundary class"""

    type: str
    name: Optional[str] = pd.Field(
        None, title="Name", description="Optional unique name for boundary."
    )


class BoundaryWithTurbulenceQuantities(Boundary, metaclass=ABCMeta):
    """Turbulence Quantities on Boundaries"""

    turbulence_quantities: Optional[TurbulenceQuantitiesType] = pd.Field(
        alias="turbulenceQuantities"
    )


class NoSlipWall(Boundary):
    """No slip wall boundary"""

    type: Literal["NoSlipWall"] = pd.Field("NoSlipWall", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")
    velocity_type: Optional[Literal["absolute", "relative"]] = pd.Field(
        default="relative", alias="velocityType"
    )


class SlipWall(Boundary):
    """Slip wall boundary"""

    type: Literal["SlipWall"] = pd.Field("SlipWall", const=True)


class RiemannInvariant(Boundary):
    """Riemann Invariant boundary"""

    type: Literal["RiemannInvariant"] = pd.Field("RiemannInvariant", const=True)


class FreestreamBoundary(BoundaryWithTurbulenceQuantities):
    """Freestream boundary"""

    type: Literal["Freestream"] = pd.Field("Freestream", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")
    velocity_type: Optional[Literal["absolute", "relative"]] = pd.Field(
        default="relative", alias="velocityType"
    )


class IsothermalWall(Boundary):
    """IsothermalWall boundary"""

    type: Literal["IsothermalWall"] = pd.Field("IsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(
        alias="Temperature", options=["Value", "Expression"]
    )
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class HeatFluxWall(Boundary):
    """:class:`HeatFluxWall` class for specifying heat flux wall boundaries

    Parameters
    ----------
    heatFlux : float
        Heat flux at the wall.

    velocity: BoundaryVelocityType
        (Optional) Velocity of the wall. If not specified, the boundary is stationary.

    Returns
    -------
    :class:`HeatFluxWall`
        An instance of the component class HeatFluxWall.

    Example
    -------
    >>> heatFluxWall = HeatFluxWall(heatFlux=-0.01, velocity=(0, 0, 0))
    """

    type: Literal["HeatFluxWall"] = pd.Field("HeatFluxWall", const=True)
    heat_flux: Union[float, StrictStr] = pd.Field(alias="heatFlux", options=["Value", "Expression"])
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="velocity")


class SubsonicOutflowPressure(BoundaryWithTurbulenceQuantities):
    """SubsonicOutflowPressure boundary"""

    type: Literal["SubsonicOutflowPressure"] = pd.Field("SubsonicOutflowPressure", const=True)
    static_pressure_ratio: PositiveFloat = pd.Field(alias="staticPressureRatio")


class SubsonicOutflowMach(BoundaryWithTurbulenceQuantities):
    """SubsonicOutflowMach boundary"""

    type: Literal["SubsonicOutflowMach"] = pd.Field("SubsonicOutflowMach", const=True)
    Mach: PositiveFloat = pd.Field(alias="MachNumber")


class SubsonicInflow(BoundaryWithTurbulenceQuantities):
    """SubsonicInflow boundary"""

    type: Literal["SubsonicInflow"] = pd.Field("SubsonicInflow", const=True)
    total_pressure_ratio: PositiveFloat = pd.Field(alias="totalPressureRatio")
    total_temperature_ratio: PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")
    velocity_direction: Optional[BoundaryAxisType] = pd.Field(alias="velocityDirection")


class SupersonicInflow(Boundary):
    """:class:`SupersonicInflow` class for specifying the full fluid state at supersonic inflow boundaries

    Parameters
    ----------
    total_temperature_ratio : PositiveFloat
        Ratio of total temperature to static temperature at the inlet.

    total_pressure_ratio: PositiveFloat
        Ratio of the total pressure to static pressure at the inlet.

    static_pressure_ratio: PositiveFloat
        Ratio of the inlet static pressure to the freestream static pressure. Default freestream static pressure in
        Flow360 = 1.0/gamma.

    velocity_direction: BoundaryAxisType
        (Optional) 3-array of either float values or string expressions. Unit vector which specifies the direction
        of the incoming flow. If not specified, the boundary patch normal is used to specify direction.

    Returns
    -------
    :class:`SupersonicInflow`
        An instance of the component class SupersonicInflow.

    Example
    -------
    >>> supersonicInflow = SupersonicInflow(
        totalTemperatureRatio=2.1,
        totalPressureRatio=3.0,
        staticPressureRatio=1.2
    )
    """

    type: Literal["SupersonicInflow"] = pd.Field("SupersonicInflow", const=True)
    total_temperature_ratio: PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    total_pressure_ratio: PositiveFloat = pd.Field(alias="totalPressureRatio")
    static_pressure_ratio: PositiveFloat = pd.Field(alias="staticPressureRatio")
    velocity_direction: Optional[BoundaryAxisType] = pd.Field(alias="velocityDirection")


class SlidingInterfaceBoundary(Boundary):
    """:class: `SlidingInterface` boundary"""

    type: Literal["SlidingInterface"] = pd.Field("SlidingInterface", const=True)


class WallFunction(Boundary):
    """:class: `WallFunction` boundary"""

    type: Literal["WallFunction"] = pd.Field("WallFunction", const=True)


class MassInflow(BoundaryWithTurbulenceQuantities):
    """:class: `MassInflow` boundary"""

    type: Literal["MassInflow"] = pd.Field("MassInflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class MassOutflow(BoundaryWithTurbulenceQuantities):
    """:class: `MassOutflow` boundary"""

    type: Literal["MassOutflow"] = pd.Field("MassOutflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class SolidIsothermalWall(Boundary):
    """:class: `SolidIsothermalWall` boundary"""

    type: Literal["SolidIsothermalWall"] = pd.Field("SolidIsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(
        alias="Temperature", options=["Value", "Expression"]
    )


class SolidAdiabaticWall(Boundary):
    """:class: `SolidAdiabaticWall` boundary"""

    type: Literal["SolidAdiabaticWall"] = pd.Field("SolidAdiabaticWall", const=True)


class TranslationallyPeriodic(Boundary):
    """:class: `TranslationallyPeriodic` boundary"""

    type: Literal["TranslationallyPeriodic"] = pd.Field("TranslationallyPeriodic", const=True)
    paired_patch_name: Optional[str] = pd.Field(alias="pairedPatchName")
    translation_vector: Optional[Vector] = pd.Field(alias="translationVector")


class RotationallyPeriodic(Boundary):
    """:class: `RotationallyPeriodic` boundary"""

    type: Literal["RotationallyPeriodic"] = pd.Field("RotationallyPeriodic", const=True)
    paired_patch_name: Optional[str] = pd.Field(alias="pairedPatchName")
    axis_of_rotation: Optional[Vector] = pd.Field(alias="axisOfRotation")
    theta_radians: Optional[float] = pd.Field(alias="thetaRadians")


class SymmetryPlane(Boundary):
    """Symmetry plane boundary - normal gradients forced to be zero"""

    type: Literal["SymmetryPlane"] = pd.Field("SymmetryPlane", const=True)


class VelocityInflow(BoundaryWithTurbulenceQuantities):
    """Inflow velocity for incompressible solver"""

    type: Literal["VelocityInflow"] = pd.Field("VelocityInflow", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class PressureOutflow(BoundaryWithTurbulenceQuantities):
    """Outflow pressure for incompressible solver"""

    type: Literal["PressureOutflow"] = pd.Field("PressureOutflow", const=True)
    static_pressure: Optional[PressureType] = pd.Field(alias="staticPressure")
    length_scale_factor: Optional[PositiveFloat] = pd.Field(alias="lengthScaleFactor")


BoundaryType = Union[
    NoSlipWall,
    SlipWall,
    FreestreamBoundary,
    IsothermalWall,
    HeatFluxWall,
    SubsonicOutflowPressure,
    SubsonicOutflowMach,
    SubsonicInflow,
    SupersonicInflow,
    SlidingInterfaceBoundary,
    WallFunction,
    MassInflow,
    MassOutflow,
    SolidIsothermalWall,
    SolidAdiabaticWall,
    TranslationallyPeriodic,
    RotationallyPeriodic,
    SymmetryPlane,
    RiemannInvariant,
    VelocityInflow,
    PressureOutflow,
]
