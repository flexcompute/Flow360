""" Classes related to material definitions representing different media contained within the simulation """

from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...


class Material(Flow360BaseModel):
    # contains models of getting properites, for example US standard atmosphere model
    name: str = pd.Field()
    dynamic_viscosity: float = pd.Field()


class Air(Material):
    name: Literal["air"] = pd.Field(frozen=True)
    dynamic_viscosity: float = pd.Field(18.03e-6, frozen=True)
