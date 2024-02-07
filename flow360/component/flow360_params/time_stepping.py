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

from ..types import (
    Axis,
    Coordinate,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Vector,
)
from .params_base import DeprecatedAlias, Flow360BaseModel
from .unit_system import TimeType


class RampCFL(Flow360BaseModel):
    """
    Ramp CFL for time stepping component
    """

    type: Literal["ramp"] = pd.Field("ramp", const=True)
    initial: Optional[PositiveFloat] = pd.Field(default=5)
    final: Optional[PositiveFloat] = pd.Field(default=200)
    ramp_steps: Optional[int] = pd.Field(alias="rampSteps", default=40)

    askedForDefault = True  # User wants to use default

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.askedForDefault = not bool(kwargs)

    class Config(Flow360BaseModel.Config):
        exclude_on_flow360_export = ["askedForDefault"]

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady CFL settings
        """
        return cls(initial=1, final=1e6, ramp_steps=30)  ## Unknown souce of the values


class AdaptiveCFL(Flow360BaseModel):
    """
    Adaptive CFL for time stepping component
    """

    type: Literal["adaptive"] = pd.Field("adaptive", const=True)
    min: Optional[PositiveFloat] = pd.Field(default=0.1)
    max: Optional[PositiveFloat] = pd.Field(default=1e4)
    max_relative_change: Optional[PositiveFloat] = pd.Field(alias="maxRelativeChange", default=1)
    convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
        alias="convergenceLimitingFactor", default=0.25
    )

    askedForDefault = True  # User wants to use default

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.askedForDefault = not bool(kwargs)

    class Config(Flow360BaseModel.Config):
        exclude_on_flow360_export = ["askedForDefault"]

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady CFL settings
        """
        return cls(max=1e6, convergence_limiting_factor=1.0, max_relative_change=50)


# pylint: disable=E0213
class BaseTimeStepping(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for time stepping component
    """

    max_pseudo_steps: Optional[pd.conint(gt=0, le=100000)] = pd.Field(2000, alias="maxPseudoSteps")

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
        displayed="CFL", options=["Ramp CFL", "Adaptive CFL"], default=RampCFL()
    )


class UnsteadyTimeStepping(BaseTimeStepping):
    """
    Unsteady time stepping component
    """

    model_type: Literal["Unsteady"] = pd.Field("Unsteady", alias="modelType", const=True)
    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    time_step_size: Optional[TimeType.Positive] = pd.Field(alias="timeStepSize")
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field(
        displayed="CFL", options=["Ramp CFL", "Adaptive CFL"]
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.CFL is None or (isinstance(self.CFL, RampCFL) and self.CFL.askedForDefault):
            print("Setting default unsteady RAMP CFL")
            self.CFL = RampCFL.default_unsteady()
        elif isinstance(self.CFL, AdaptiveCFL) and self.CFL.askedForDefault:
            self.CFL = AdaptiveCFL.default_unsteady()


TimeStepping = Union[SteadyTimeStepping, UnsteadyTimeStepping]
