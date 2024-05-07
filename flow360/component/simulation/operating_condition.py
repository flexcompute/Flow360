from typing import Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

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
    # TODO: Units!!
    ##Note: Did not specify fileds as optional cause I am sure we will forget to specify proper defaults if it is optional and have "None" related errors.
    pressure: float = pd.Field(111)
    altitude: float = pd.Field(111)
    velocity: float = pd.Field(111)
    Mach: float = pd.Field(111)
    alpha: float = pd.Field(0)
    beta: float = pd.Field(0)
    temperature: float = pd.Field(111)
    Reynolds: float = (
        111,
        pd.Field(),
    )  # Note: Is this possible to just compute Reynolds from user inputs?

    initial_condition: tuple[str, str, str] = pd.Field(("None", "None", "None"))


class InternalFlowOperatingConditions(Flow360BaseModel):
    pressure_difference: float = pd.Field()
    velocity: float = pd.Field()


class SolidOperatingConditions(Flow360BaseModel): ...


OperatingConditionTypes = Union[
    ExternalFlowOperatingConditions, InternalFlowOperatingConditions, SolidOperatingConditions
]
