"""
Contains basically only boundary conditions for now. In future we can add new models like 2D equations.
"""

# pylint: disable=too-many-lines

import logging
import warnings
from abc import ABCMeta
from typing import Annotated, Literal, Union

import pydantic as pd
import unyt as u

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.expression import StringExpression
from flow360_schema.framework.physical_dimensions import (
    AbsoluteTemperature,
    AngularVelocity,
    InverseArea,
    InverseLength,
    Length,
)
from flow360_schema.framework.physical_dimensions import HeatFlux as HeatFluxDim
from flow360_schema.framework.physical_dimensions import MassFlowRate as MassFlowRateDim
from flow360_schema.framework.physical_dimensions import Pressure as PressureDim
from flow360_schema.framework.single_attribute_base import SingleAttributeModel
from flow360_schema.framework.unique_list import UniqueItemList
from flow360_schema.models.simulation.framework.updater_utils import deprecation_reminder
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    GhostSurfacePair,
    MirroredSurface,
    Surface,
    SurfacePair,
    WindTunnelGhostSurface,
)
from flow360_schema.models.simulation.models.turbulence_quantities import (
    TurbulenceQuantitiesType,
)
from flow360_schema.models.simulation.operating_condition.operating_condition import (
    VelocityVectorType,
)
from flow360_schema.models.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_field_validator,
)
from flow360_schema.models.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
    check_deleted_surface_pair,
    remap_symmetric_ghost_entity,
    validate_entity_list_surface_existence,
)

logger = logging.getLogger(__name__)


class BoundaryBase(Flow360BaseModel, metaclass=ABCMeta):
    """Boundary base"""

    type: str = pd.Field()
    entities: EntityList[Surface, MirroredSurface] = pd.Field(
        alias="surfaces",
        description="List of boundaries with boundary condition imposed.",
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def remap_symmetric_to_user_name(cls, value, param_info: ParamsValidationInfo):
        """Remap 'symmetric' ghost entity to user's symmetry surface name for UDF backward compat."""
        return remap_symmetric_ghost_entity(value, param_info)

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)


class BoundaryBaseWithTurbulenceQuantities(BoundaryBase, metaclass=ABCMeta):
    """Boundary base with turbulence quantities"""

    turbulence_quantities: TurbulenceQuantitiesType | None = pd.Field(
        None,
        description="The turbulence related quantities definition." + "See :func:`TurbulenceQuantities` documentation.",
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
    value: StringExpression | HeatFluxDim.Float64 = pd.Field(description="The heat flux value.")


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
    value: StringExpression | AbsoluteTemperature.Float64 = pd.Field(description="The temperature value.")


class TotalPressure(Flow360BaseModel):
    """
    :class:`TotalPressure` class to specify the total pressure for `Inflow`
    boundary condition via :py:attr:`Inflow.spec`.

    Example
    -------

    - Using a constant value:

      >>> fl.TotalPressure(
      ...     value = 1.04e6 * fl.u.Pa,
      ... )

    - Using an expression (nondimensionalized by Flow360 pressure unit, rho * a^2):

      >>> fl.TotalPressure(
      ...     value = "pow(1.0+0.2*pow(0.1*(1.0-y*y),2.0),1.4/0.4) / 1.4",
      ... )

    ====
    """

    type_name: Literal["TotalPressure"] = pd.Field("TotalPressure", frozen=True)
    value: StringExpression | PressureDim.PositiveFloat64 = pd.Field(
        description="The total pressure value. When a string expression is supplied the value"
        + " needs to be nondimensionalized by the Flow360 pressure unit (rho_inf * a_inf^2)."
    )


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
    value: PressureDim.PositiveFloat64 = pd.Field(description="The static pressure value.")


class SlaterPorousBleed(Flow360BaseModel):
    """
    :class:`SlaterPorousBleed` is a no-slip wall model which prescribes a normal
    velocity at the surface as a function of the surface pressure and density according
    to the model of John Slater.

    Example
    -------
    - Specify a static pressure of 1.01e6 Pascals at the slater bleed boundary, and
      set the porosity of the surface to 0.4 (40%).

    >>> fl.SlaterPorousBleed(static_pressure=1.01e6 * fl.u.Pa, porosity=0.4, activation_step=200)

    ====
    """

    type_name: Literal["SlaterPorousBleed"] = pd.Field("SlaterPorousBleed", frozen=True)
    static_pressure: PressureDim.PositiveFloat64 = pd.Field(description="The static pressure value.")
    porosity: float = pd.Field(gt=0, le=1, description="The porosity of the bleed region.")
    activation_step: pd.PositiveInt | None = pd.Field(
        None, description="Pseudo step at which to start applying the SlaterPorousBleedModel."
    )


class MassFlowRate(Flow360BaseModel):
    """
    :class:`MassFlowRate` class to specify the mass flow rate for `Inflow` or `Outflow`
    boundary condition via :py:attr:`Inflow.spec`/:py:attr:`Outflow.spec`.

    Example
    -------

    >>> fl.MassFlowRate(
    ...     value = 123 * fl.u.lb / fl.u.s,
    ...     ramp_steps = 100,
    ... )

    ====
    """

    type_name: Literal["MassFlowRate"] = pd.Field("MassFlowRate", frozen=True)
    value: MassFlowRateDim.NonNegativeFloat64 = pd.Field(description="The mass flow rate.")
    ramp_steps: pd.PositiveInt | None = pd.Field(
        None,
        description="Number of pseudo steps before reaching :py:attr:`MassFlowRate.value` within 1 physical step.",
    )


class Supersonic(Flow360BaseModel):
    """
    :class:`Supersonic` class to specify the supersonic conditions for `Inflow`.

    Example
    -------

    >>> fl.Supersonic(
    ...     total_pressure = 7.90e6 * fl.u.Pa,
    ...     static_pressure = 1.01e6 * fl.u.Pa,
    ... )

    """

    type_name: Literal["Supersonic"] = pd.Field("Supersonic", frozen=True)
    total_pressure: PressureDim.PositiveFloat64 = pd.Field(description="The total pressure.")
    static_pressure: PressureDim.PositiveFloat64 = pd.Field(description="The static pressure.")


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
    axis_of_rotation: Axis | None = pd.Field(None)


class WallRotation(Flow360BaseModel):
    """
    :class:`WallRotation` class to specify the rotational velocity model for the `Wall` boundary condition.

    The wall rotation model prescribes a rotational motion at the wall by defining a center of rotation,
    an axis about which the wall rotates, and an angular velocity. This model can be used to simulate
    rotating components or surfaces in a flow simulation.

    Example
    -------
    >>> fl.Wall(
    ...     entities=volume_mesh["fluid/wall"],
    ...     velocity=fl.WallRotation(
    ...         axis=(0, 0, 1),
    ...         center=(1, 2, 3) * u.m,
    ...         angular_velocity=100 * u.rpm
    ...     ),
    ...     use_wall_function=True,
    ... )

    ====
    """

    center: Length.Vector3 = pd.Field(description="The center of rotation")
    axis: Axis = pd.Field(description="The axis of rotation.")
    angular_velocity: AngularVelocity.Float64 = pd.Field(description="The value of the angular velocity.")
    type_name: Literal["WallRotation"] = pd.Field("WallRotation", frozen=True)
    private_attribute_circle_mode: dict | None = pd.Field(None)


WallVelocityModelTypes = Annotated[SlaterPorousBleed | WallRotation, pd.Field(discriminator="type_name")]
WALL_VELOCITY_MODEL_ADAPTER = pd.TypeAdapter(WallVelocityModelTypes)


class WallFunction(Flow360BaseModel):
    """
    :class:`WallFunction` specifies the wall function model to use on a :class:`Wall` boundary.

    Example
    -------

    - Default boundary-layer wall function:

      >>> fl.Wall(
      ...     entities=volume_mesh["fluid/wall"],
      ...     use_wall_function=fl.WallFunction(),
      ... )

    - Inner-layer wall model:

      >>> fl.Wall(
      ...     entities=volume_mesh["fluid/wall"],
      ...     use_wall_function=fl.WallFunction(wall_function_type="InnerLayer"),
      ... )

    ====
    """

    wall_function_type: Literal["BoundaryLayer", "InnerLayer"] = pd.Field(
        "BoundaryLayer",
        description="Type of wall function model. "
        + "'BoundaryLayer' uses integral flat plate boundary layer theory to predict wall shear stress. "
        + "It performs well across all y+ ranges. "
        + "'InnerLayer' uses the inner layer behavior of the turbulent boundary layer, "
        + "offering better accuracy for y+ values in the log layer and below.",
    )


class Wall(BoundaryBase):
    """
    :class:`Wall` class defines the wall boundary condition based on the inputs.

    Example
    -------

    - :code:`Wall` with default wall function (BoundaryLayer) and prescribed velocity:

      >>> fl.Wall(
      ...     entities=geometry["wall_function"],
      ...     velocity = ["min(0.2, 0.2 + 0.2*y/0.5)", "0", "0.1*y/0.5"],
      ...     use_wall_function=fl.WallFunction(),
      ... )

    - :code:`Wall` with inner-layer wall function:

      >>> fl.Wall(
      ...     entities=volume_mesh["8"],
      ...     use_wall_function=fl.WallFunction(wall_function_type="InnerLayer"),
      ... )

    - :code:`Wall` with wall function and wall rotation:

      >>> fl.Wall(
      ...     entities=volume_mesh["8"],
      ...     velocity=WallRotation(
      ...       axis=(0, 0, 1),
      ...       center=(1, 2, 3) * u.m,
      ...       angular_velocity=100 * u.rpm
      ...     ),
      ...     use_wall_function=fl.WallFunction(),
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

    - Define Slater no-slip bleed model on entities
      with the naming pattern :code:`"fluid/SlaterBoundary-*"`:

      >>> fl.Wall(
      ...     entities=volume_mesh["fluid/SlaterBoundary-*"],
      ...     velocity=fl.SlaterPorousBleed(
      ...         static_pressure=1.01e6 * fl.u.Pa, porosity=0.4, activation_step=200
      ...     ),
      ... )

    - Define roughness height on entities
      with the naming pattern :code:`"fluid/Roughness-*"`:

      >>> fl.Wall(
      ...     entities=volume_mesh["fluid/Roughness-*"],
      ...     roughness_height=0.1 * fl.u.mm,
      ... )

    ====
    """

    name: str | None = pd.Field("Wall", description="Name of the `Wall` boundary condition.")
    type: Literal["Wall"] = pd.Field("Wall", frozen=True)
    use_wall_function: WallFunction | None = pd.Field(
        None,
        description="Wall function configuration. Set to :class:`WallFunction` to enable "
        + "wall functions. The default wall function type is ``'BoundaryLayer'``. "
        + "Set to ``None`` to disable wall functions (no-slip wall).",
    )

    velocity: WallVelocityModelTypes | VelocityVectorType | None = pd.Field(
        None, description="Prescribe a velocity or the velocity model on the wall."
    )

    heat_spec: HeatFlux | Temperature = pd.Field(
        HeatFlux(0 * u.W / u.m**2),
        discriminator="type_name",
        description="Specify the heat flux or temperature at the `Wall` boundary.",
    )
    roughness_height: Length.NonNegativeFloat64 = pd.Field(
        0 * u.m,
        description="Equivalent sand grain roughness height. Available only to `Fluid` zone boundaries.",
    )
    private_attribute_dict: dict | None = pd.Field(None)

    entities: EntityList[Surface, MirroredSurface, WindTunnelGhostSurface] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the `Wall` boundary condition imposed.",
    )

    @pd.field_validator("use_wall_function", mode="before")
    @classmethod
    def _normalize_wall_function(cls, value):
        """Handle backward-compatible bool inputs for use_wall_function."""
        if value is True:
            logger.warning(
                "Passing a bool to `use_wall_function` is deprecated. "
                "Use `use_wall_function=WallFunction()` instead of `True`."
            )
            return WallFunction()
        if value is False:
            logger.warning(
                "Passing a bool to `use_wall_function` is deprecated. "
                "Use `use_wall_function=None` instead of `False`."
            )
            return None
        return value

    @pd.field_validator("velocity", mode="before")
    @classmethod
    def _normalize_velocity(cls, value):
        if isinstance(value, dict) and value.get("type_name") is not None:
            return WALL_VELOCITY_MODEL_ADAPTER.validate_python(value)
        return value

    @pd.model_validator(mode="after")
    def check_wall_function_conflict(self):
        """Check no setting is conflicting with the usage of wall function"""
        if self.use_wall_function is None:
            return self
        if isinstance(self.velocity, SlaterPorousBleed):
            raise ValueError(f"Using `{type(self.velocity).__name__}` with wall function is not supported currently.")
        return self

    @contextual_field_validator("heat_spec", mode="after")
    @classmethod
    def _ensure_adiabatic_wall_for_liquid(cls, value, param_info: ParamsValidationInfo):
        """Allow only adiabatic wall when liquid operating condition is used"""
        if param_info.using_liquid_as_material is False:
            return value
        if isinstance(value, HeatFlux) and value.value == 0 * u.W / u.m**2:
            return value
        raise ValueError("Only adiabatic wall is allowed when using liquid as simulation material.")

    @contextual_field_validator("velocity", mode="after")
    @classmethod
    def _disable_expression_for_liquid(cls, value, param_info: ParamsValidationInfo):
        if param_info.using_liquid_as_material is False:
            return value

        if isinstance(value, tuple):
            if isinstance(value[0], str) and isinstance(value[1], str) and isinstance(value[2], str):
                raise ValueError("Expression cannot be used when using liquid as simulation material.")
        return value


class Freestream(BoundaryBaseWithTurbulenceQuantities):
    """
    :class:`Freestream` defines the freestream boundary condition.

    Example
    -------

    - Define freestream boundary condition with velocity expression and boundaries from the volume mesh:

      >>> fl.Freestream(
      ...     surfaces=[volume_mesh["blk-1/freestream-part1"],
      ...               volume_mesh["blk-1/freestream-part2"]],
      ...     velocity = ["min(0.2, 0.2 + 0.2*y/0.5)", "0", "0.1*y/0.5"]
      ... )

    - Define freestream boundary condition with turbulence quantities and automated farfield:

      >>> auto_farfield = fl.AutomatedFarfield()
      ... fl.Freestream(
      ...     entities=[auto_farfield.farfield],
      ...     turbulence_quantities= fl.TurbulenceQuantities(
      ...         modified_viscosity_ratio=10,
      ...     )
      ... )

    ====
    """

    name: str | None = pd.Field("Freestream", description="Name of the `Freestream` boundary condition.")
    type: Literal["Freestream"] = pd.Field("Freestream", frozen=True)
    velocity: VelocityVectorType | None = pd.Field(
        None,
        description="The default values are set according to the "
        + ":py:attr:`AerospaceCondition.alpha` and :py:attr:`AerospaceCondition.beta` angles. "
        + "Optionally, an expression for each of the velocity components can be specified.",
    )
    entities: EntityList[
        Surface,
        MirroredSurface,
        GhostSurface,
        WindTunnelGhostSurface,
        GhostSphere,
        GhostCircularPlane,
    ] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the `Freestream` boundary condition imposed.",
    )

    @contextual_field_validator("velocity", mode="after")
    @classmethod
    def _disable_expression_for_liquid(cls, value, param_info: ParamsValidationInfo):
        if param_info.using_liquid_as_material is False:
            return value

        if isinstance(value, tuple):
            if isinstance(value[0], str) and isinstance(value[1], str) and isinstance(value[2], str):
                raise ValueError("Expression cannot be used when using liquid as simulation material.")
        return value


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
      ...     spec=fl.MassFlowRate(value = 123 * fl.u.lb / fl.u.s)
      ... )

    ====
    """

    name: str | None = pd.Field("Outflow", description="Name of the `Outflow` boundary condition.")
    type: Literal["Outflow"] = pd.Field("Outflow", frozen=True)
    spec: Pressure | MassFlowRate | Mach = pd.Field(
        discriminator="type_name",
        description="Specify the static pressure, mass flow rate, or Mach number parameters at"
        + " the `Outflow` boundary.",
    )
    entities: EntityList[Surface, MirroredSurface, WindTunnelGhostSurface] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the `Outflow` boundary condition imposed.",
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
      ...     spec=fl.TotalPressure(
      ...         value = 1.028e6 * fl.u.Pa,
      ...     ),
      ...     velocity_direction = (1, 0, 0),
      ... )

    - Define inflow boundary condition with mass flow rate:

      >>> fl.Inflow(
      ...     entities=[volume_mesh["fluid/inflow"]],
      ...     total_temperature=300 * fl.u.K,
      ...     spec=fl.MassFlowRate(
      ...         value = 123 * fl.u.lb / fl.u.s,
      ...         ramp_steps = 10,
      ...     ),
      ...     velocity_direction = (1, 0, 0),
      ... )

    - Define inflow boundary condition with turbulence quantities:

      >>> fl.Inflow(
      ...     entities=[volume_mesh["fluid/inflow"]],
      ...     turbulence_quantities=fl.TurbulenceQuantities(
      ...         turbulent_kinetic_energy=2.312e-3 * fl.u.m **2 / fl.u.s**2,
      ...         specific_dissipation_rate= 1020 / fl.u.s,
      ...     )
      ... )

    - Define inflow boundary condition with expressions for spatially varying total temperature and total pressure:

      >>> fl.Inflow(
      ...     entities=[volumeMesh["fluid/inflow"]],
      ...     total_temperature="1.0+0.2*pow(0.1*(1.0-y*y),2.0)",
      ...     velocity_direction=(1.0, 0.0, 0.0),
      ...     spec=fl.TotalPressure(
      ...         value="pow(1.0+0.2*pow(0.1*(1.0-y*y),2.0),1.4/0.4)",
      ...     ),
      ... )

    ====
    """

    name: str | None = pd.Field("Inflow", description="Name of the `Inflow` boundary condition.")
    type: Literal["Inflow"] = pd.Field("Inflow", frozen=True)
    total_temperature: StringExpression | AbsoluteTemperature.Float64 = pd.Field(
        description="Specify the total temperature at the `Inflow` boundary."
        + " When a string expression is supplied the value"
        + " needs to nondimensionalized by the temperature defined in `operating_condition`."
    )
    spec: TotalPressure | MassFlowRate | Supersonic = pd.Field(
        discriminator="type_name",
        description="Specify additional conditions at the `Inflow` boundary.",
    )
    velocity_direction: Axis | None = pd.Field(
        None,
        description="Direction of the incoming flow. Must be a unit vector pointing "
        + "into the volume. If unspecified, the direction will be normal to the surface.",
    )
    rotate_velocity_direction_with_mesh: bool = pd.Field(
        False,
        description="When True, the velocity direction vector rotates with the mesh at each "
        + "physical time step. Use this when the inflow boundary is inside a rotating zone and "
        + "the velocity direction should be specified relative to the body frame rather than the "
        + "inertial frame. Only relevant when `velocity_direction` is set.",
    )
    entities: EntityList[Surface, MirroredSurface, WindTunnelGhostSurface] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the `Inflow` boundary condition imposed.",
    )

    @pd.model_validator(mode="after")
    def _ensure_velocity_direction_for_mesh_rotation(self):
        """`rotate_velocity_direction_with_mesh` is only meaningful when `velocity_direction` is set."""
        if self.rotate_velocity_direction_with_mesh and self.velocity_direction is None:
            raise ValueError(
                "`rotate_velocity_direction_with_mesh` cannot be set when `velocity_direction` is not specified."
            )
        return self


class SlipWall(BoundaryBase):
    """:class:`SlipWall` class defines the :code:`SlipWall` boundary condition.

    Example
    -------

    Define :code:`SlipWall` boundary condition for entities with the naming pattern:

    :code:`"*/slipWall"` in the volume mesh.

      >>> fl.SlipWall(entities=volume_mesh["*/slipWall"]

    - Define :code:`SlipWall` boundary condition with automated farfield symmetry plane boundaries:

      >>> auto_farfield = fl.AutomatedFarfield()
      >>> fl.SlipWall(
      ...     entities=[auto_farfield.symmetry_planes],
      ...     turbulence_quantities= fl.TurbulenceQuantities(
      ...         modified_viscosity_ratio=10,
      ...     )
      ... )

    ====
    """

    name: str | None = pd.Field("Slip wall", description="Name of the `SlipWall` boundary condition.")
    type: Literal["SlipWall"] = pd.Field("SlipWall", frozen=True)
    entities: EntityList[Surface, MirroredSurface, GhostSurface, WindTunnelGhostSurface, GhostCircularPlane] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the :code:`SlipWall` boundary condition imposed.",
    )


class SymmetryPlane(BoundaryBase):
    """
    :class:`SymmetryPlane` defines the symmetric boundary condition.
    It is similar to :class:`SlipWall`, but the normal gradient of scalar quantities
    are forced to be zero on the symmetry plane. **Only planar surfaces are supported.**

    Example
    -------

    >>> fl.SymmetryPlane(entities=volume_mesh["fluid/symmetry"])

    - Define `SymmetryPlane` boundary condition with automated farfield symmetry plane boundaries:

      >>> auto_farfield = fl.AutomatedFarfield()
      >>> fl.SymmetryPlane(
      ...     entities=[auto_farfield.symmetry_planes],
      ... )

    ====
    """

    name: str | None = pd.Field("Symmetry", description="Name of the `SymmetryPlane` boundary condition.")
    type: Literal["SymmetryPlane"] = pd.Field("SymmetryPlane", frozen=True)
    entities: EntityList[Surface, MirroredSurface, GhostSurface, GhostCircularPlane] = pd.Field(
        alias="surfaces",
        description="List of boundaries with the `SymmetryPlane` boundary condition imposed.",
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

    name: str | None = pd.Field("Periodic", description="Name of the `Periodic` boundary condition.")
    type: Literal["Periodic"] = pd.Field("Periodic", frozen=True)
    entity_pairs: UniqueItemList[SurfacePair | GhostSurfacePair] = pd.Field(
        alias="surface_pairs",
        description="List of matching pairs of :class:`~flow360.Surface` or `~flow360.GhostSurface`. ",
    )
    spec: Translational | Rotational = pd.Field(
        discriminator="type_name",
        description="Define the type of periodic boundary condition (translational/rotational) "
        + "via :class:`Translational`/:class:`Rotational`.",
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @contextual_field_validator("entity_pairs", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        for surface_pair in value.items:
            check_deleted_surface_pair(surface_pair, param_info)
        return value

    @contextual_field_validator("entity_pairs", mode="after")
    @classmethod
    def _ensure_quasi_3d_periodic_when_using_ghost_surface(cls, value, param_info: ParamsValidationInfo):
        """
        When using ghost surface pairs, ensure the farfield type is quasi-3d-periodic.
        """
        for surface_pair in value.items:
            if isinstance(surface_pair, GhostSurfacePair):
                if param_info.farfield_method != "quasi-3d-periodic":
                    raise ValueError("Farfield type must be 'quasi-3d-periodic' when using GhostSurfacePair.")
        return value


class PorousJump(Flow360BaseModel):
    """
    :class:`PorousJump` defines the Porous Jump boundary condition.

    Provide a flat list of every face that should be a porous-jump boundary
    via ``surfaces``. Each face must belong to a multizone interface; the
    donor/receiver pairing is recovered from the mesh metadata downstream.

    Example
    -------

      >>> fl.PorousJump(
      ...     surfaces=[
      ...         volume_mesh["blk-1/Interface-blk-2"],
      ...         volume_mesh["blk-2/Interface-blk-1"],
      ...         volume_mesh["blk-1/Interface-blk-3"],
      ...         volume_mesh["blk-3/Interface-blk-1"],
      ...     ],
      ...    darcy_coefficient = 1e6 / fl.u.m **2,
      ...    forchheimer_coefficient = 1 / fl.u.m,
      ...    thickness = 1 * fl.u.m,
      ... )

    .. deprecated::
        The ``surface_pairs=[(A, B), ...]`` pair form is accepted for one
        more minor release and is rewritten internally into the flat
        ``surfaces`` form. New code should use ``surfaces=`` directly.

    ====
    """

    name: str | None = pd.Field("PorousJump", description="Name of the `PorousJump` boundary condition.")
    type: Literal["PorousJump"] = pd.Field("PorousJump", frozen=True)
    entities: EntityList[Surface, MirroredSurface] = pd.Field(
        alias="surfaces",
        description=(
            "Flat list of boundaries that form porous-jump interfaces. "
            "Each face must belong to a multizone interface; the donor/"
            "receiver pairing is recovered from mesh metadata downstream."
        ),
    )
    darcy_coefficient: InverseArea.Float64 = pd.Field(
        description="Darcy coefficient of the porous media model which determines the scaling of the "
        + "viscous loss term. The value defines the coefficient for the axis normal "
        + "to the surface."
    )
    forchheimer_coefficient: InverseLength.Float64 = pd.Field(
        description="Forchheimer coefficient of the porous media model which determines "
        + "the scaling of the inertial loss term."
    )
    thickness: Length.Float64 = pd.Field(description="Thickness of the thin porous media on the surface")
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    # Target version is the highest patch within the current minor so the
    # validator survives any 25.10.x patch but raises on the 25.11.0 bump.
    @pd.model_validator(mode="before")
    @classmethod
    @deprecation_reminder("25.10.999")
    def _expand_legacy_pair_input(cls, data):
        """Rewrite legacy ``surface_pairs`` / ``entity_pairs`` input into the
        canonical flat ``surfaces`` form. Mutual exclusion with ``surfaces``
        is enforced here. Emits a :class:`DeprecationWarning` whenever the
        legacy form is used.
        """
        if not isinstance(data, dict):
            return data

        legacy_key = next(
            (k for k in ("surface_pairs", "entity_pairs") if data.get(k) is not None),
            None,
        )
        if legacy_key is None:
            return data

        if any(data.get(k) is not None for k in ("surfaces", "entities")):
            raise ValueError("PorousJump: provide either `surfaces` or `surface_pairs`, not both.")

        pairs = data.pop(legacy_key)
        if isinstance(pairs, dict) and "items" in pairs:
            pair_items = pairs["items"]
        elif isinstance(pairs, list):
            pair_items = pairs
        else:
            raise ValueError(f"PorousJump: `{legacy_key}` must be a list of surface pairs.")

        flat: list = []
        for item in pair_items:
            if isinstance(item, dict) and "pair" in item:
                flat.extend(item["pair"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                flat.extend(item)
            elif hasattr(item, "pair"):
                flat.extend(item.pair)
            else:
                raise ValueError(f"PorousJump: unrecognized `{legacy_key}` element {item!r}.")

        data["surfaces"] = flat
        deprecation_message = (
            "PorousJump: `surface_pairs` is deprecated; pass `surfaces=[A, B, ...]` "
            "as a flat list of boundaries instead. Donor/receiver pairing is "
            "recovered from mesh metadata downstream."
        )
        # logger.warning surfaces via the client's bridge handler regardless of
        # Python's warning filters; warnings.warn keeps the standard hook for
        # tooling (pytest.warns, CI deprecation scanners).
        logger.warning(deprecation_message)
        warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)
        return data

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_porous_jump_interfaces(cls, value, param_info: ParamsValidationInfo):
        """Validate that every surface will exist after meshing and either is,
        or will become, an interface.

        A surface is accepted if it is already an interface OR will become one
        after meshing (it is a boundary of a CustomVolume, on the farfield
        enclosed set, or is a dual-belonging face).
        """
        expanded = param_info.expand_entity_list(value)
        check_deleted_surface_in_entity_list(expanded, param_info)

        cv_boundary_ids: set = set()
        for cv_info in param_info.to_be_generated_custom_volumes.values():
            cv_boundary_ids |= cv_info.get("boundary_surface_ids", set())
        enclosed_ids = set(param_info.farfield_enclosed_entities or {})
        dual_ids = set(param_info.farfield_cv_dual_belonging_ids or set())

        for surface in expanded:
            if not isinstance(surface, Surface):
                continue
            if surface.private_attribute_is_interface:
                continue
            sid = surface.private_attribute_id
            if sid in cv_boundary_ids or sid in enclosed_ids or sid in dual_ids:
                continue
            raise ValueError(f"Boundary `{surface.name}` is not an interface")

        return value


SurfaceModelTypes = Union[
    Wall,
    SlipWall,
    Freestream,
    Outflow,
    Inflow,
    Periodic,
    SymmetryPlane,
    PorousJump,
]
