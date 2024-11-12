"""Time stepping setting for simulation"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import TimeType


def _apply_default_to_none(original, default):
    for field_name, value in original.model_dump().items():
        if value is None:
            setattr(original, field_name, default.model_dump()[field_name])
    return original


class RampCFL(Flow360BaseModel):
    """
    :class:`RampCFL` class for the Ramp CFL setting of time stepping.
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
        description="Number of pseudo steps before reaching :paramref:`RampCFL.final` within 1 physical step.",
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
        + "In unsteady simulations, the value of :paramref:`AdaptiveCFL.max_relative_change` "
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


class BaseTimeStepping(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for time stepping component
    """

    order_of_accuracy: Literal[1, 2] = pd.Field(2)


class Steady(BaseTimeStepping):
    """
    :class:`Steady` class for specifying steady simulation.
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
            return values  # will be handeled by default value
        cfl_input = values["CFL"]
        if isinstance(cfl_input, AdaptiveCFL):
            cfl_input = _apply_default_to_none(cfl_input, AdaptiveCFL.default_steady())
        elif isinstance(cfl_input, RampCFL):
            cfl_input = _apply_default_to_none(cfl_input, RampCFL.default_steady())
        return values


class Unsteady(BaseTimeStepping):
    """
    :class:`Unsteady` class for specifying unsteady simulation.
    """

    type_name: Literal["Unsteady"] = pd.Field("Unsteady", frozen=True)
    max_pseudo_steps: int = pd.Field(
        100, gt=0, le=100000, description="Maximum pseudo steps within one physical step."
    )
    steps: pd.PositiveInt = pd.Field(description="Number of physical steps.")
    # pylint: disable=no-member
    step_size: TimeType.Positive = pd.Field(description="Time step size in physical step marching,")
    # pylint: disable=duplicate-code
    CFL: Union[RampCFL, AdaptiveCFL] = pd.Field(
        default=AdaptiveCFL.default_unsteady(),
        description="CFL settings within each physical step.",
    )

    @pd.model_validator(mode="before")
    @classmethod
    def set_default_cfl(cls, values):
        """
        Populate CFL's None fields with default
        """
        if "CFL" not in values:
            return values  # will be handeled by default value
        cfl_input = values["CFL"]
        if isinstance(cfl_input, AdaptiveCFL):
            cfl_input = _apply_default_to_none(cfl_input, AdaptiveCFL.default_unsteady())
        elif isinstance(cfl_input, RampCFL):
            cfl_input = _apply_default_to_none(cfl_input, RampCFL.default_unsteady())
        return values
