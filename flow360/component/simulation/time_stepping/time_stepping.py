from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


def _apply_default_to_none(original, default):
    for field_name in original.__fields__:
        value = getattr(original, field_name)
        if value is None:
            setattr(original, field_name, default.dict()[field_name])
    return original


class RampCFL(Flow360BaseModel):
    """
    Ramp CFL for time stepping component
    """

    type: Literal["ramp"] = pd.Field("ramp", frozen=True)
    initial: Optional[pd.PositiveFloat] = pd.Field(None)
    final: Optional[pd.PositiveFloat] = pd.Field(None)
    ramp_steps: Optional[int] = pd.Field(None)

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
    Adaptive CFL for time stepping component
    """

    type: Literal["adaptive"] = pd.Field("adaptive", frozen=True)
    min: pd.PositiveFloat = pd.Field(default=0.1)
    max: Optional[pd.PositiveFloat] = pd.Field(None)
    max_relative_change: Optional[pd.PositiveFloat] = pd.Field(None)
    convergence_limiting_factor: Optional[pd.PositiveFloat] = pd.Field(None)

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

    max_pseudo_steps: int = pd.Field(2000, gt=0, le=100000)
    order_of_accuracy: Literal[1, 2] = pd.Field(2)
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field(None)

    #  TODO:
    # # pylint: disable=arguments-differ
    # def to_solver(self, params, **kwargs) -> BaseTimeStepping:
    #     """
    #     returns configuration object in flow360 units system
    #     """
    #     return super().to_solver(params, **kwargs)


class SteadyTimeStepping(BaseTimeStepping):
    """
    Steady time stepping component
    """

    model_type: Literal["Steady"] = pd.Field("Steady", frozen=True)
    CFL: Union[RampCFL, AdaptiveCFL] = pd.Field(
        default=AdaptiveCFL.default_steady(),
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


class UnsteadyTimeStepping(BaseTimeStepping):
    """
    Unsteady time stepping component
    """

    model_type: Literal["Unsteady"] = pd.Field("Unsteady", frozen=True)
    physical_steps: pd.PositiveInt = pd.Field()
    time_step_size: TimeType.Positive = pd.Field()
    CFL: Union[RampCFL, AdaptiveCFL] = pd.Field(
        default=AdaptiveCFL.default_unsteady(),
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
