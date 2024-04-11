from typing import Union

import pydantic as pd

from flow360.component.flow360_params.params_base import Flow360BaseModel

"""
    Defines all the operating conditions for different physics.
    The type of operating condition has to match its volume type.
    Operating conditions defines:
    1. The physical (non-geometrical) reference values for the problem.
    2. The initial condition for the problem.

    TODO:
    1. What types of operation conditions do we need?
"""


class ExternalFlowOperatingConditions(Flow360BaseModel):

    pressure: float = pd.Field()
    altitude: float = pd.Field()
    velocity: float = pd.Field()
    mach: float = pd.Field()
    alpha: float = pd.Field()
    beta: float = pd.Field()

    initial_condition: tuple[str, str, str] = pd.Field()


class InternalFlowOperatingConditions(Flow360BaseModel):
    pressure_difference: float = pd.Field()
    velocity: float = pd.Field()


class SolidOperatingConditions(Flow360BaseModel): ...


OperatingConditionTypes = Union[
    ExternalFlowOperatingConditions, InternalFlowOperatingConditions, SolidOperatingConditions
]
