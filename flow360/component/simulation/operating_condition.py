from typing import Optional, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

"""
    Defines all the operating conditions for different physics.
    The type of operating condition has to match its volume type.
    Operating conditions defines:
    1. The physical (non-geometrical) reference values for the problem.
    2. The initial condition for the problem.

    TODO:
    1. What other types of operation conditions do we need?
"""


class TurbulenceQuantities(Flow360BaseModel):
    """PLACE HOLDER, Should be exactly the the same as `TurbulenceQuantitiesType` in current Flow360Params"""

    pass


class ExternalFlowOperatingConditions(Flow360BaseModel):
    Mach: float = pd.Field()
    alpha: float = pd.Field(0)
    beta: float = pd.Field(0)
    temperature: float = pd.Field(288.15)
    reference_velocity: Optional[float] = pd.Field()  # See U_{ref} definition in our documentation

    initial_flow_condition: Optional[tuple[str, str, str, str, str]] = pd.Field(
        ("NotImplemented", "NotImplemented", "NotImplemented")
    )
    turbulence_quantities = Optional[TurbulenceQuantities] = pd.Field()


class InternalFlowOperatingConditions(Flow360BaseModel):
    pressure_difference: float = pd.Field()
    reference_velocity: float = pd.Field()
    inlet_velocity: float = pd.Field()


class SolidOperatingConditions(Flow360BaseModel):
    initial_temperature: float = pd.Field()


OperatingConditionTypes = Union[
    ExternalFlowOperatingConditions, InternalFlowOperatingConditions, SolidOperatingConditions
]
