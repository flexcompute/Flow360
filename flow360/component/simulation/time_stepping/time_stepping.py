"""Time stepping setting for simulation"""

from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import TimeType
from flow360.component.simulation.user_code.core.types import ValueOrExpression


def _apply_default_to_none(original, default):
    for field_name, value in original.model_dump().items():
        if value is None:
            setattr(original, field_name, default.model_dump()[field_name])
    return original


class RampCFL(Flow360BaseModel):
    """
    :class:`RampCFL` class for the Ramp CFL setting of time stepping.

    Example
    -------

    >>> fl.RampCFL(initial=1, final=200, ramp_steps=200)

    ====
    """

    type: Literal["ramp"] = pd.Field("ramp", frozen=True)
    initial: Optional[pd.PositiveFloat] = pd.Field(
        None, description="Initial CFL for solving pseudo time step."
    )
    final: Optional[pd.PositiveFloat] = pd.Field(
        None, description="Final CFL for solving pseudo time step."
    )
    ramp_steps: Optional[pd.PositiveInt] = pd.Field(
        None,
        description="Number of pseudo steps before reaching :py:attr:`RampCFL.final` within 1 physical step.",
    )

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady Ramp CFL settings
        """
        return cls(initial=1, final=1e6, ramp_steps=30)

    @classmethod
    def default_steady(cls):
        """
        returns default steady Ramp CFL settings
        """
        return cls(initial=5, final=200, ramp_steps=40)


class AdaptiveCFL(Flow360BaseModel):
    """
    :class:`AdaptiveCFL` class for Adaptive CFL setting of time stepping.

    Example
    -------

    - Set up Adaptive CFL with convergence limiting factor:

      >>> fl.AdaptiveCFL(convergence_limiting_factor=0.5)

    - Set up Adaptive CFL with max relative change:

      >>> fl.AdaptiveCFL(
      ...     min=1,
      ...     max=100000,
      ...     max_relative_change=50
      ... )

    ====
    """

    type: Literal["adaptive"] = pd.Field("adaptive", frozen=True)
    min: pd.PositiveFloat = pd.Field(
        default=0.1, description="The minimum allowable value for Adaptive CFL."
    )
    max: Optional[pd.PositiveFloat] = pd.Field(
        None, description="The maximum allowable value for Adaptive CFL."
    )
    max_relative_change: Optional[pd.PositiveFloat] = pd.Field(
        None,
        description="The maximum allowable relative change of CFL (%) at each pseudo step. "
        + "In unsteady simulations, the value of :py:attr:`AdaptiveCFL.max_relative_change` "
        + "is updated automatically depending on how well the solver converges in each physical step.",
    )
    convergence_limiting_factor: Optional[pd.PositiveFloat] = pd.Field(
        None,
        description="This factor specifies the level of conservativeness when using Adaptive CFL. "
        + "Smaller values correspond to a more conservative limitation on the value of CFL.",
    )

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady Adaptive CFL settings
        """
        return cls(max=1e6, convergence_limiting_factor=1.0, max_relative_change=50)

    @classmethod
    def default_steady(cls):
        """
        returns default steady Adaptive CFL settings
        """
        return cls(max=1e4, convergence_limiting_factor=0.25, max_relative_change=1)


class Steady(Flow360BaseModel):
    """
    :class:`Steady` class for specifying steady simulation.

    Example
    -------

    >>> fl.Steady(
    ...     CFL=fl.RampCFL(initial=1, final=200, ramp_steps=200),
    ...     max_steps=6000,
    ... )

    ====

    """

    type_name: Literal["Steady"] = pd.Field("Steady", frozen=True)
    max_steps: int = pd.Field(2000, gt=0, le=100000, description="Maximum number of pseudo steps.")
    # pylint: disable=duplicate-code
    CFL: Union[RampCFL, AdaptiveCFL] = pd.Field(
        default=AdaptiveCFL.default_steady(), description="CFL settings."
    )

    @pd.model_validator(mode="before")
    @classmethod
    def set_default_cfl(cls, values):
        """
        Populate CFL's None fields with default
        """
        if "CFL" not in values:
            return values  # will be handled by default value
        cfl_input = values["CFL"]
        if isinstance(cfl_input, AdaptiveCFL):
            cfl_input = _apply_default_to_none(cfl_input, AdaptiveCFL.default_steady())
        elif isinstance(cfl_input, RampCFL):
            cfl_input = _apply_default_to_none(cfl_input, RampCFL.default_steady())
        return values


class Unsteady(Flow360BaseModel):
    """
    :class:`Unsteady` class for specifying unsteady simulation.

    Example
    -------

    >>> fl.Unsteady(
    ...     CFL=fl.AdaptiveCFL(
    ...         convergence_limiting_factor=0.5
    ...     ),
    ...     step_size=0.01 * fl.u.s,
    ...     steps=120,
    ...     max_pseudo_steps=35,
    ... )

    ====
    """

    type_name: Literal["Unsteady"] = pd.Field("Unsteady", frozen=True)
    max_pseudo_steps: int = pd.Field(
        20, gt=0, le=100000, description="Maximum pseudo steps within one physical step."
    )
    steps: pd.PositiveInt = pd.Field(description="Number of physical steps.")
    # pylint: disable=no-member
    step_size: ValueOrExpression[TimeType.Positive] = pd.Field(
        description="Time step size in physical step marching,"
    )
    # pylint: disable=duplicate-code
    CFL: Union[RampCFL, AdaptiveCFL] = pd.Field(
        default=AdaptiveCFL.default_unsteady(),
        description="CFL settings within each physical step.",
    )
    order_of_accuracy: Literal[1, 2] = pd.Field(2, description="Temporal order of accuracy.")

    @pd.model_validator(mode="before")
    @classmethod
    def set_default_cfl(cls, values):
        """
        Populate CFL's None fields with default
        """
        if "CFL" not in values:
            return values  # will be handled by default value
        cfl_input = values["CFL"]
        if isinstance(cfl_input, AdaptiveCFL):
            cfl_input = _apply_default_to_none(cfl_input, AdaptiveCFL.default_unsteady())
        elif isinstance(cfl_input, RampCFL):
            cfl_input = _apply_default_to_none(cfl_input, RampCFL.default_unsteady())
        return values
