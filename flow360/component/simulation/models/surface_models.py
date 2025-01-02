"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
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
from flow360.component.simulation.primitives import (
    GhostCircularPlane,
    GhostSurface,
    Surface,
    SurfacePair,
)
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
    via :py:attr:`Wall.heat_spec`.

    Example
    -------

    >>> fl.HeatFlux(value = 1.0 * fl.u.W/fl.u.m**2)

    ====
    """

    type_name: Literal["HeatFlux"] = pd.Field("HeatFlux", frozen=True)
    value: Union[StringExpression, HeatFluxType] = pd.Field(description="The heat flux value.")


class Temperature(SingleAttributeModel):
    """
    :class:`Temperature` class to specify the temperature for `Wall` or `Inflow`
    boundary condition via :py:attr:`Wall.heat_spec`/
    :py:attr:`Inflow.spec`.

    Example
    -------

    >>> fl.Temperature(value = 350 * fl.u.K)

    ====
    """

    type_name: Literal["Temperature"] = pd.Field("Temperature", frozen=True)
    # pylint: disable=no-member
    value: Union[StringExpression, TemperatureType.Positive] = pd.Field(
        description="The temperature value."
    )


class TotalPressure(SingleAttributeModel):
    """
    :class:`TotalPressure` class to specify the total pressure for `Inflow`
    boundary condition via :py:attr:`Inflow.spec`.

    Example
    -------

    >>> fl.TotalPressure(value = 1.04e6 * fl.u.Pa)

    ====
    """

    type_name: Literal["TotalPressure"] = pd.Field("TotalPressure", frozen=True)
    # pylint: disable=no-member
    value: PressureType.Positive = pd.Field(description="The total pressure value.")


class Pressure(SingleAttributeModel):
    """
    :class:`Pressure` class to specify the static pressure for `Outflow`
    boundary condition via :py:attr:`Outflow.spec`.

    Example
    -------

    >>> fl.Pressure(value = 1.01e6 * fl.u.Pa)

    ====
    """

    type_name: Literal["Pressure"] = pd.Field("Pressure", frozen=True)
    # pylint: disable=no-member
    value: PressureType.Positive = pd.Field(description="The static pressure value.")


class MassFlowRate(SingleAttributeModel):
    """
    :class:`MassFlowRate` class to specify the mass flow rate for `Inflow` or `Outflow`
    boundary condition via :py:attr:`Inflow.spec`/:py:attr:`Outflow.spec`.

    Example
    -------

    >>> fl.MassFlowRate(value = 123 * fl.u.lb / fl.u.s)

    ====
    """

    type_name: Literal["MassFlowRate"] = pd.Field("MassFlowRate", frozen=True)
    # pylint: disable=no-member
    value: MassFlowRateType.NonNegative = pd.Field(description="The mass flow rate.")


class Mach(SingleAttributeModel):
    """
    :class:`Mach` class to specify Mach number for the `Inflow`
    boundary condition via :py:attr:`Inflow.spec`.

    Example
    -------

    >>> fl.Mach(value = 0.5)

    ====
    """

    type_name: Literal["Mach"] = pd.Field("Mach", frozen=True)
    value: pd.NonNegativeFloat = pd.Field(description="The Mach number.")


class Translational(Flow360BaseModel):
    """
    :class:`Translational` class to specify translational periodic
    boundary condition via :py:attr:`Periodic.spec`.

    """

    type_name: Literal["Translational"] = pd.Field("Translational", frozen=True)


class Rotational(Flow360BaseModel):
    """
    :class:`Rotational` class to specify rotational periodic
    boundary condition via :py:attr:`Periodic.spec`.
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
    :class:`Wall` class defines the wall boundary condition based on the inputs.

    Example
    -------

    - :code:`Wall` with wall function and prescribed velocity:

      >>> fl.Wall(
      ...     entities=geometry["wall_function"],
      ...     velocity = ["min(0.2, 0.2 + 0.2*y/0.5)", "0", "0.1*y/0.5"],
      ...     use_wall_function=True,
      ... )

      >>> fl.Wall(
      ...     entities=volume_mesh["8"],
      ...     velocity = (
      ...         f"{OMEGA[1]} * (z - {CENTER[2]}) - {OMEGA[2]} * (y - {CENTER[1]})",
      ...         f"{OMEGA[2]} * (x - {CENTER[0]}) - {OMEGA[0]} * (z - {CENTER[2]})",
      ...         f"{OMEGA[0]} * (y - {CENTER[1]}) - {OMEGA[1]} * (x - {CENTER[0]})",
      ...     ),
      ...     use_wall_function=True,
      ... )

    - Define isothermal wall boundary condition on entities
      with the naming pattern :code:`"fluid/isothermal-*"`:

      >>> fl.Wall(
      ...     entities=volume_mesh["fluid/isothermal-*"],
      ...     heat_spec=fl.Temperature(350 * fl.u.K),
      ... )

    - Define isoflux wall boundary condition on entities
      with the naming pattern :code:`"solid/isoflux-*"`:

      >>> fl.Wall(
      ...     entities=volume_mesh["solid/isoflux-*"],
      ...     heat_spec=fl.HeatFlux(1.0 * fl.u.W/fl.u.m**2),
      ... )

    ====
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
    # pylint: disable=no-member
    heat_spec: Union[HeatFlux, Temperature] = pd.Field(
        HeatFlux(0 * u.W / u.m**2),
        discriminator="type_name",
        description="Specify the heat flux or temperature at the `Wall` boundary.",
    )


class Freestream(BoundaryBaseWithTurbulenceQuantities):
    """
    :class:`Freestream` defines the freestream boundary condition.

    Example
    -------

    - Define freestream boundary condition with velocity expression:

      >>> fl.Freestream(
      ...     surfaces=[volume_mesh["blk-1/zblocks"],
      ...               volume_mesh["blk-1/xblocks"]],
      ...     velocity = ["min(0.2, 0.2 + 0.2*y/0.5)", "0", "0.1*y/0.5"]
      ... )

    - Define freestream boundary condition with turbulence quantities:

      >>> fl.Freestream(
      ...     entities=[volume_mesh['freestream']],
      ...     turbulence_quantities= fl.TurbulenceQuantities(
      ...         modified_viscosity_ratio=10,
      ...     )
      ... )

    ====
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Freestream` boundary condition.")
    type: Literal["Freestream"] = pd.Field("Freestream", frozen=True)
    velocity: Optional[VelocityVectorType] = pd.Field(
        None,
        description="The default values are set according to the "
        + ":py:attr:`AerospaceCondition.alpha` and :py:attr:`AerospaceCondition.beta` angles. "
        + "Optionally, an expression for each of the velocity components can be specified.",
    )
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `Freestream` boundary condition imposed.",
    )


class Outflow(BoundaryBase):
    """
    :class:`Outflow` defines the outflow boundary condition based on the input :py:attr:`spec`.

    Example
    -------
    - Define outflow boundary condition with pressure:

      >>> fl.Outflow(
      ...     surfaces=volume_mesh["fluid/outlet"],
      ...     spec=fl.Pressure(value = 0.99e6 * fl.u.Pa)
      ... )

    - Define outflow boundary condition with Mach number:

      >>> fl.Outflow(
      ...     surfaces=volume_mesh["fluid/outlet"],
      ...     spec=fl.Mach(value = 0.2)
      ... )

    - Define outflow boundary condition with mass flow rate:

      >>> fl.Outflow(
      ...     surfaces=volume_mesh["fluid/outlet"],
      ...     spec=fl.MassFlowRate(123 * fl.u.lb / fl.u.s)
      ... )

    ====
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Outflow` boundary condition.")
    type: Literal["Outflow"] = pd.Field("Outflow", frozen=True)
    spec: Union[Pressure, MassFlowRate, Mach] = pd.Field(
        discriminator="type_name",
        description="Specify the static pressure, mass flow rate or Mach number at the `Outflow` boundary.",
    )


class Inflow(BoundaryBaseWithTurbulenceQuantities):
    """
    :class:`Inflow` defines the inflow boundary condition based on the input :py:attr:`spec`.

    Example
    -------

    - Define inflow boundary condition with pressure:

      >>> fl.Inflow(
      ...     entities=[geometry["inflow"]],
      ...     total_temperature=300 * fl.u.K,
      ...     spec=fl.TotalPressure(1.028e6 * fl.u.Pa),
      ... )

    - Define inflow boundary condition with mass flow rate:

      >>> fl.Inflow(
      ...     entities=[volume_mesh["fluid/inflow"]],
      ...     total_temperature=300 * fl.u.K,
      ...     spec=fl.MassFlowRate(123 * fl.u.lb / fl.u.s),
      ... )

    - Define inflow boundary condition with turbulence quantities:

      >>> fl.Inflow(
      ...     entities=[volume_mesh["fluid/inflow"]],
      ...     turbulence_quantities= fl.TurbulenceQuantities(
      ...         turbulent_kinetic_energy=2.312e-3 * fl.u.m **2 / fl.u.s**2,
      ...         specific_dissipation_rate= 1020 / fl.u.s,
      ...     )
      ... )

    ====
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

    @pd.model_validator(mode="after")
    def _check_velocity_direction_in_subsonic_inflow(self):
        """
        Check to ensure velocity_direction can only be specified for Subsonic Inflow Boundary.
        """
        if self.velocity_direction and not isinstance(self.spec, TotalPressure):
            raise ValueError("The velocity_direction can only be specified for Subsonic Inflow.")
        return self


class SlipWall(BoundaryBase):
    """:class:`SlipWall` class defines the :code:`SlipWall` boundary condition.

    Example
    -------

    Define :code:`SlipWall` boundary condition for entities with the naming pattern
    :code:`"*/slipWall"` in the volume mesh.

    >>> fl.SlipWall(entities=volume_mesh["*/slipWall"]

    ====
    """

    name: Optional[str] = pd.Field(None, description="Name of the `SlipWall` boundary condition.")
    type: Literal["SlipWall"] = pd.Field("SlipWall", frozen=True)
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `SlipWall` boundary condition imposed.",
    )


class SymmetryPlane(BoundaryBase):
    """
    :class:`SymmetryPlane` defines the symmetric boundary condition.
    It is similar to :class:`SlipWall`, but the normal gradient of scalar quantities
    are forced to be zero on the symmetry plane. **Only planar surfaces are supported.**

    Example
    -------

    >>> fl.SymmetryPlane(entities=volume_mesh["fluid/symmetry"])

    ====
    """

    name: Optional[str] = pd.Field(
        None, description="Name of the `SymmetryPlane` boundary condition."
    )
    type: Literal["SymmetryPlane"] = pd.Field("SymmetryPlane", frozen=True)
    entities: EntityList[Surface, GhostSurface, GhostCircularPlane] = pd.Field(
        alias="surfaces",
        description="A list of :class:`Surface` entities with "
        + "the `SymmetryPlane` boundary condition imposed.",
    )


class Periodic(Flow360BaseModel):
    """
    :class:`Periodic` defines the translational or rotational periodic boundary condition.

    Example
    -------

    - Define a translationally periodic boundary condition using :class:`Translational`:

      >>> fl.Periodic(
      ...     surface_pairs=[
      ...         (volume_mesh["VOLUME/BOTTOM"], volume_mesh["VOLUME/TOP"]),
      ...         (volume_mesh["VOLUME/RIGHT"], volume_mesh["VOLUME/LEFT"]),
      ...     ],
      ...     spec=fl.Translational(),
      ... )

    - Define a rotationally periodic boundary condition using :class:`Rotational`:

      >>> fl.Periodic(
      ...     surface_pairs=[(volume_mesh["VOLUME/PERIODIC-1"],
      ...         volume_mesh["VOLUME/PERIODIC-2"])],
      ...     spec=fl.Rotational()
      ... )

    ====
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
