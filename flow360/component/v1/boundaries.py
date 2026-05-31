"""
Boundaries parameters
"""

from __future__ import annotations

from abc import ABCMeta
from typing import Literal, Optional, Tuple, Union

import pydantic.v1 as pd
from pydantic.v1 import StrictStr

from flow360.component.types import Axis, Vector
from flow360.component.utils import process_expressions
from flow360.component.v1.params_base import Flow360BaseModel
from flow360.component.v1.turbulence_quantities import TurbulenceQuantitiesType
from flow360.component.v1.unit_system import PressureType, VelocityType

BoundaryVelocityType = Union[VelocityType.Vector, Tuple[StrictStr, StrictStr, StrictStr]]
BoundaryAxisType = Union[Axis, Tuple[StrictStr, StrictStr, StrictStr]]

UpwindPhiBCTypeNames = {
    "Freestream",
    "RiemannInvariant",
    "SubsonicOutflowPressure",
    "PressureOutflow",
    "SubsonicOutflowMach",
    "SubsonicInflow",
    "MassOutflow",
    "MassInflow",
}


def _check_velocity_is_expression(input_velocity):
    if not isinstance(input_velocity, tuple) or len(input_velocity) != 3:
        return False
    return all(isinstance(element, str) for element in input_velocity)


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
        alias="turbulenceQuantities", discriminator="model_type"
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> BoundaryWithTurbulenceQuantities:
        """
        Apply freestream turbulence quantities to applicable boudnaries
        """

        if params.freestream.turbulence_quantities is not None:
            if self.type in UpwindPhiBCTypeNames and self.turbulence_quantities is None:
                self.turbulence_quantities = params.freestream.turbulence_quantities
        return super().to_solver(params, **kwargs)


class NoSlipWall(Boundary):
    """No slip wall boundary"""

    type: Literal["NoSlipWall"] = pd.Field("NoSlipWall", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)


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

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)


class IsothermalWall(Boundary):
    """IsothermalWall boundary"""

    type: Literal["IsothermalWall"] = pd.Field("IsothermalWall", const=True)
    temperature: Union[pd.PositiveFloat, StrictStr] = pd.Field(
        alias="Temperature", options=["Value", "Expression"]
    )
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)
    _processed_temperature = pd.validator("temperature", allow_reuse=True)(process_expressions)


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
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)
    _processed_heat_flux = pd.validator("heat_flux", allow_reuse=True)(process_expressions)


class SubsonicOutflowPressure(BoundaryWithTurbulenceQuantities):
    """SubsonicOutflowPressure boundary"""

    type: Literal["SubsonicOutflowPressure"] = pd.Field("SubsonicOutflowPressure", const=True)
    static_pressure_ratio: pd.PositiveFloat = pd.Field(alias="staticPressureRatio")


class SubsonicOutflowMach(BoundaryWithTurbulenceQuantities):
    """SubsonicOutflowMach boundary"""

    type: Literal["SubsonicOutflowMach"] = pd.Field("SubsonicOutflowMach", const=True)
    Mach: pd.PositiveFloat = pd.Field(alias="MachNumber")


class SubsonicInflow(BoundaryWithTurbulenceQuantities):
    """SubsonicInflow boundary"""

    type: Literal["SubsonicInflow"] = pd.Field("SubsonicInflow", const=True)
    total_pressure_ratio: pd.PositiveFloat = pd.Field(alias="totalPressureRatio")
    total_temperature_ratio: pd.PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    ramp_steps: Optional[pd.PositiveInt] = pd.Field(alias="rampSteps")
    velocity_direction: Optional[BoundaryAxisType] = pd.Field(alias="velocityDirection")


class SupersonicInflow(Boundary):
    """:class:`SupersonicInflow` class for specifying the full fluid state at supersonic inflow boundaries

    Parameters
    ----------
    total_temperature_ratio :pd.PositiveFloat
        Ratio of total temperature to static temperature at the inlet.

    total_pressure_ratio:pd.PositiveFloat
        Ratio of the total pressure to static pressure at the inlet.

    static_pressure_ratio:pd.PositiveFloat
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
    total_temperature_ratio: pd.PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    total_pressure_ratio: pd.PositiveFloat = pd.Field(alias="totalPressureRatio")
    static_pressure_ratio: pd.PositiveFloat = pd.Field(alias="staticPressureRatio")
    velocity_direction: Optional[BoundaryAxisType] = pd.Field(alias="velocityDirection")


class SlidingInterfaceBoundary(Boundary):
    """:class: `SlidingInterface` boundary"""

    type: Literal["SlidingInterface"] = pd.Field("SlidingInterface", const=True)


class WallFunction(Boundary):
    """:class: `WallFunction` boundary"""

    type: Literal["WallFunction"] = pd.Field("WallFunction", const=True)

    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)


class MassInflow(BoundaryWithTurbulenceQuantities):
    """:class: `MassInflow` boundary"""

    type: Literal["MassInflow"] = pd.Field("MassInflow", const=True)
    mass_flow_rate: pd.PositiveFloat = pd.Field(alias="massFlowRate")
    total_temperature_ratio: pd.PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    ramp_steps: Optional[pd.PositiveInt] = pd.Field(alias="rampSteps")


class MassOutflow(BoundaryWithTurbulenceQuantities):
    """:class: `MassOutflow` boundary"""

    type: Literal["MassOutflow"] = pd.Field("MassOutflow", const=True)
    mass_flow_rate: pd.PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[pd.PositiveInt] = pd.Field(alias="rampSteps")


class SolidIsothermalWall(Boundary):
    """:class: `SolidIsothermalWall` boundary"""

    type: Literal["SolidIsothermalWall"] = pd.Field("SolidIsothermalWall", const=True)
    temperature: Union[pd.PositiveFloat, StrictStr] = pd.Field(
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

    _processed_velocity = pd.validator("velocity", allow_reuse=True)(process_expressions)


class PressureOutflow(BoundaryWithTurbulenceQuantities):
    """Outflow pressure for incompressible solver"""

    type: Literal["PressureOutflow"] = pd.Field("PressureOutflow", const=True)
    static_pressure: Optional[PressureType] = pd.Field(alias="staticPressure")
    length_scale_factor: Optional[pd.PositiveFloat] = pd.Field(alias="lengthScaleFactor")


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
