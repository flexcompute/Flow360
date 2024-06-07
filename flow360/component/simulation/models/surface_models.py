"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.single_attribute_base import (
    SingleAttributeModel,
)
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.operating_condition import VelocityVectorType
from flow360.component.simulation.primitives import Surface, SurfacePair
from flow360.component.simulation.unit_system import (
    HeatFluxType,
    MassFlowRateType,
    PressureType,
    TemperatureType,
)

# TODO: Warning: Pydantic V1 import
from flow360.component.types import Axis

from .turbulence_quantities import TurbulenceQuantitiesType


class BoundaryBase(Flow360BaseModel, metaclass=ABCMeta):
    type: str = pd.Field()
    entities: EntityList[Surface] = pd.Field(alias="surfaces")


class BoundaryBaseWithTurbulenceQuantities(BoundaryBase, metaclass=ABCMeta):
    turbulence_quantities: Optional[TurbulenceQuantitiesType] = pd.Field(None)


class HeatFlux(SingleAttributeModel):
    value: Union[HeatFluxType, pd.StrictStr] = pd.Field()


class Temperature(SingleAttributeModel):
    value: Union[TemperatureType.Positive, pd.StrictStr] = pd.Field()


class TotalPressure(SingleAttributeModel):
    value: PressureType.Positive = pd.Field()


class Pressure(SingleAttributeModel):
    value: PressureType.Positive = pd.Field()


class MassFlowRate(SingleAttributeModel):
    value: MassFlowRateType.NonNegative = pd.Field()


class Mach(SingleAttributeModel):
    value: pd.NonNegativeFloat = pd.Field()


class Wall(BoundaryBase):
    """Replace Flow360Param:
    - NoSlipWall
    - IsothermalWall
    - HeatFluxWall
    - WallFunction
    - SolidIsothermalWall
    - SolidAdiabaticWall
    """

    type: Literal["Wall"] = pd.Field("Wall", frozen=True)
    use_wall_function: bool = pd.Field(False)
    velocity: Optional[VelocityVectorType] = pd.Field(None)
    velocity_type: Literal["absolute", "relative"] = pd.Field("relative")
    heat_spec: Optional[Union[HeatFlux, Temperature]] = pd.Field(None)


class Freestream(BoundaryBaseWithTurbulenceQuantities):
    type: Literal["Freestream"] = pd.Field("Freestream", frozen=True)
    velocity: Optional[VelocityVectorType] = pd.Field(None)
    velocity_type: Literal["absolute", "relative"] = pd.Field("relative")


class Outflow(BoundaryBase):
    """Replace Flow360Param:
    - SubsonicOutflowPressure
    - SubsonicOutflowMach
    - MassOutflow
    """

    type: Literal["Outflow"] = pd.Field("Outflow", frozen=True)
    spec: Union[Pressure, MassFlowRate, Mach] = pd.Field()


class Inflow(BoundaryBaseWithTurbulenceQuantities):
    """Replace Flow360Param:
    - SubsonicInflow
    - MassInflow
    """

    type: Literal["Inflow"] = pd.Field("Inflow", frozen=True)
    total_temperature: TemperatureType.Positive = pd.Field()
    velocity_direction: Optional[Axis] = pd.Field(None)
    spec: Union[TotalPressure, MassFlowRate] = pd.Field()


class SlipWall(BoundaryBase):
    type: Literal["SlipWall"] = pd.Field("SlipWall", frozen=True)


class SymmetryPlane(BoundaryBase):
    type: Literal["SymmetryPlane"] = pd.Field("SymmetryPlane", frozen=True)


class Translational(Flow360BaseModel):
    type: Literal["Translational"] = pd.Field("Translational", frozen=True)


class Rotational(Flow360BaseModel):
    type: Literal["Rotational"] = pd.Field("Rotational", frozen=True)
    # pylint: disable=fixme
    # TODO: Maybe we need more precision when serializeing this one?
    axis_of_rotation: Optional[Axis] = pd.Field(None)


class Periodic(Flow360BaseModel):
    type: Literal["Periodic"] = pd.Field("Periodic", frozen=True)
    entity_pairs: UniqueItemList[SurfacePair] = pd.Field(alias="surface_pairs")
    spec: Union[Translational, Rotational] = pd.Field(discriminator="type")


SurfaceModelTypes = Union[
    Wall,
    SlipWall,
    Freestream,
    Outflow,
    Inflow,
    Periodic,
    SymmetryPlane,
]
