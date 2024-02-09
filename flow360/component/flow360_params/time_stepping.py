"""
Time stepping parameters
"""

# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

from abc import ABCMeta
from typing import Dict, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..types import PositiveFloat, PositiveInt
from .params_base import DeprecatedAlias, Flow360BaseModel
from .unit_system import TimeType


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

    type: Literal["ramp"] = pd.Field("ramp", const=True)
    initial: Optional[PositiveFloat] = pd.Field()
    final: Optional[PositiveFloat] = pd.Field()
    ramp_steps: Optional[int] = pd.Field(alias="rampSteps")

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

    type: Literal["adaptive"] = pd.Field("adaptive", const=True)
    min: Optional[PositiveFloat] = pd.Field(default=0.1)
    max: Optional[PositiveFloat] = pd.Field()
    max_relative_change: Optional[PositiveFloat] = pd.Field(alias="maxRelativeChange")
    convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
        alias="convergenceLimitingFactor"
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


# pylint: disable=E0213
class BaseTimeStepping(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for time stepping component
    """

    max_pseudo_steps: Optional[pd.conint(gt=0, le=100000)] = pd.Field(2000, alias="maxPseudoSteps")
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2, alias="orderOfAccuracy")
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field(
        displayed="CFL", options=["Ramp CFL", "Adaptive CFL"]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> BaseTimeStepping:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="physical_steps", deprecated="maxPhysicalSteps")]
        exclude_on_flow360_export = ["model_type"]


class SteadyTimeStepping(BaseTimeStepping):
    """
    Steady time stepping component
    """

    model_type: Literal["Steady"] = pd.Field("Steady", alias="modelType", const=True)
    physical_steps: Literal[1] = pd.Field(1, alias="physicalSteps", const=True)
    time_step_size: Literal["inf"] = pd.Field("inf", alias="timeStepSize", const=True)
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field(
        displayed="CFL",
        options=["Ramp CFL", "Adaptive CFL"],
        default=AdaptiveCFL.default_steady(),
    )

    @pd.root_validator(pre=True)
    def set_default_cfl(cls, values):
        """
        Populate CFL's None fields with default
        """
        if "CFL" not in values:
            return values  # will be handeled by default
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

    model_type: Literal["Unsteady"] = pd.Field("Unsteady", alias="modelType", const=True)
    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    time_step_size: Optional[TimeType.Positive] = pd.Field(alias="timeStepSize")
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field(
        displayed="CFL",
        options=["Ramp CFL", "Adaptive CFL"],
        default=AdaptiveCFL.default_unsteady(),
    )

    @pd.root_validator(pre=True)
    def set_default_cfl(cls, values):
        """
        Populate CFL's None fields with default
        """
        if "CFL" not in values:
            return values  # will be handeled by default
        cfl_input = values["CFL"]
        if isinstance(cfl_input, AdaptiveCFL):
            cfl_input = _apply_default_to_none(cfl_input, AdaptiveCFL.default_unsteady())
        elif isinstance(cfl_input, RampCFL):
            cfl_input = _apply_default_to_none(cfl_input, RampCFL.default_unsteady())
        return values


TimeStepping = Union[SteadyTimeStepping, UnsteadyTimeStepping]
