"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.framework.single_attribute_base import (
    SingleAttributeModel,
)
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantitiesType,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    VelocityVectorType,
)
from flow360.component.simulation.primitives import GhostSurface, Surface, SurfacePair
from flow360.component.simulation.unit_system import (
    HeatFluxType,
    MassFlowRateType,
    PressureType,
    TemperatureType,
)

# pylint: disable=fixme
# TODO: Warning: Pydantic V1 import
from flow360.component.types import Axis


class BoundaryBase(Flow360BaseModel, metaclass=ABCMeta):
    """Boundary base"""

    type: str = pd.Field()
    entities: EntityList[Surface] = pd.Field(alias="surfaces")


class BoundaryBaseWithTurbulenceQuantities(BoundaryBase, metaclass=ABCMeta):
    """Boundary base with turbulence quantities"""

    turbulence_quantities: Optional[TurbulenceQuantitiesType] = pd.Field(
        None,
        description="The turbulence related quantities definition."
        + "See :func:`TurbulenceQuantities` documentation.",
    )


class HeatFlux(SingleAttributeModel):
    """
    :class:`HeatFlux` class to specify the heat flux for `Wall` boundary condition
    via :paramref:`Wall.heat_spec`.
    """

    type_name: Literal["HeatFlux"] = pd.Field("HeatFlux", frozen=True)
    value: Union[HeatFluxType, StringExpression] = pd.Field(description="The heat flux value.")


class Temperature(SingleAttributeModel):
    """
    :class:`Temperature` class to specify the temperature for `Wall` or `Inflow`
    boundary condition via :paramref:`Wall.heat_spec`/
    :paramref:`Inflow.spec`.
    """

    type_name: Literal["Temperature"] = pd.Field("Temperature", frozen=True)
    # pylint: disable=no-member
    value: Union[TemperatureType.Positive, StringExpression] = pd.Field(
        description="The temperature value."
    )


class TotalPressure(SingleAttributeModel):
    """
    :class:`TotalPressure` class to specify the total pressure for `Inflow`
    boundary condition via :paramref:`Inflow.spec`.
    """

    type_name: Literal["TotalPressure"] = pd.Field("TotalPressure", frozen=True)
    # pylint: disable=no-member
    value: PressureType.Positive = pd.Field(description="The total pressure value.")


class Pressure(SingleAttributeModel):
    """
    :class:`Pressure` class to specify the pressure for `Outflow`
    boundary condition via :paramref:`Outflow.spec`.
    """

    type_name: Literal["Pressure"] = pd.Field("Pressure", frozen=True)
    # pylint: disable=no-member
    value: PressureType.Positive = pd.Field(description="The pressure value.")


class MassFlowRate(SingleAttributeModel):
    """
    :class:`MassFlowRate` class to specify the mass flow rate for `Inflow` or `Outflow`
    boundary condition via :paramref:`Inflow.spec`/:paramref:`Outflow.spec`.
    """

    type_name: Literal["MassFlowRate"] = pd.Field("MassFlowRate", frozen=True)
    # pylint: disable=no-member
    value: MassFlowRateType.NonNegative = pd.Field(description="The mass flow rate.")


class Mach(SingleAttributeModel):
    """
    :class:`Mach` class to specify Mach number for the `Inflow`
    boundary condition via :paramref:`Inflow.spec`.
    """

    type_name: Literal["Mach"] = pd.Field("Mach", frozen=True)
    value: pd.NonNegativeFloat = pd.Field(description="The Mach number.")


class Translational(Flow360BaseModel):
    """
    :class:`Translational` class to specify translational periodic
    boundary condition via :paramref:`Periodic.spec`.
    """

    type_name: Literal["Translational"] = pd.Field("Translational", frozen=True)


class Rotational(Flow360BaseModel):
    """
    :class:`Rotational` class to specify rotational periodic
    boundary condition via :paramref:`Periodic.spec`.
    """

    type_name: Literal["Rotational"] = pd.Field("Rotational", frozen=True)
    # pylint: disable=fixme
    # TODO: Maybe we need more precision when serializeing this one?
    axis_of_rotation: Optional[Axis] = pd.Field(None)


##########################################
############# Surface models #############
##########################################


class Wall(BoundaryBase):
    """
    :class:`Wall` class defines the Wall boundary conditions below based on the input:
        - NoSlipWall
        - IsothermalWall
        - HeatFluxWall
        - WallFunction
        - SolidIsothermalWall
        - SolidAdiabaticWall
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Wall` boundary condition.")
    type: Literal["Wall"] = pd.Field("Wall", frozen=True)
    use_wall_function: bool = pd.Field(
        False,
        description="Specify if use wall functions to estimate the velocity field "
        + "close to the solid boundaries.",
    )
    velocity: Optional[VelocityVectorType] = pd.Field(
        None, description="Prescribe a tangential velocity on the wall."
    )
    heat_spec: Optional[Union[HeatFlux, Temperature]] = pd.Field(
        None,
        discriminator="type_name",
        description="Specify the heat flux or temperature at the `Wall` boundary.",
    )


class Freestream(BoundaryBaseWithTurbulenceQuantities):
    """
    :class:`Freestream` defines the Freestream condition.
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Freestream` boundary condition.")
    type: Literal["Freestream"] = pd.Field("Freestream", frozen=True)
    velocity: Optional[VelocityVectorType] = pd.Field(
        None,
        description="The default values are set according to the "
        + ":paramref:`AerospaceCondition.alpha` and :paramref:`AerospaceCondition.beta` angles. "
        + "Optionally, an expression for each of the velocity components can be specified.",
    )
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `Freestream` boundary condition imposed.",
    )


class Outflow(BoundaryBase):
    """
    :class:`Outflow` defines the Outflow boundary conditions below based on the input:
        - SubsonicOutflowPressure
        - SubsonicOutflowMach
        - MassOutflow
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Outflow` boundary condition.")
    type: Literal["Outflow"] = pd.Field("Outflow", frozen=True)
    spec: Union[Pressure, MassFlowRate, Mach] = pd.Field(
        discriminator="type_name",
        description="Specify the static pressure, mass flow rate or Mach number at the `Outflow` boundary.",
    )


class Inflow(BoundaryBaseWithTurbulenceQuantities):
    """
    :class:`Inflow` defines the Inflow boundary condition below based on the input:
        - SubsonicInflow
        - MassInflow
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Inflow` boundary condition.")
    type: Literal["Inflow"] = pd.Field("Inflow", frozen=True)
    # pylint: disable=no-member
    total_temperature: TemperatureType.Positive = pd.Field(
        description="Specify the total temperature at the `Inflow` boundary."
    )
    velocity_direction: Optional[Axis] = pd.Field(
        None,
        description=" Direction of the incoming flow. Must be a unit vector pointing "
        + "into the volume. If unspecified, the direction will be normal to the surface.",
    )
    spec: Union[TotalPressure, MassFlowRate] = pd.Field(
        discriminator="type_name",
        description="Specify the total pressure or the mass flow rate at the `Inflow` boundary.",
    )


class SlipWall(BoundaryBase):
    """:class:`SlipWall` class defines the SlipWall boundary condition."""

    name: Optional[str] = pd.Field(None, description="Name of the `SlipWall` boundary condition.")
    type: Literal["SlipWall"] = pd.Field("SlipWall", frozen=True)
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `SlipWall` boundary condition imposed.",
    )


class SymmetryPlane(BoundaryBase):
    """
    :class:`SymmetryPlane` defines the `SymmetryPlane` boundary condition.
    It is similar to :class:`SlipWall`, but the normal gradient of scalar quantities
    are forced to be zero on the symmetry plane. Only planar surfaces are supported.
    """

    name: Optional[str] = pd.Field(
        None, description="Name of the `SymmetryPlane` boundary condition."
    )
    type: Literal["SymmetryPlane"] = pd.Field("SymmetryPlane", frozen=True)
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `SymmetryPlane` boundary condition imposed.",
    )


class Periodic(Flow360BaseModel):
    """
    :class:`Periodic` defines the translational or rotational periodic boundary condition.
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Periodic` boundary condition.")
    type: Literal["Periodic"] = pd.Field("Periodic", frozen=True)
    entity_pairs: UniqueItemList[SurfacePair] = pd.Field(
        alias="surface_pairs", description="List of matching pairs of :class:`~flow360.Surface`. "
    )
    spec: Union[Translational, Rotational] = pd.Field(
        discriminator="type_name",
        description="Define the type of periodic boundary condition (translational/rotational) "
        + "via :class:`Translational`/:class:`Rotational`.",
    )


SurfaceModelTypes = Union[
    Wall,
    SlipWall,
    Freestream,
    Outflow,
    Inflow,
    Periodic,
    SymmetryPlane,
]
