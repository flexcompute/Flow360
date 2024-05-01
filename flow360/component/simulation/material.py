from typing import Literal

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...


class Material(Flow360BaseModel):
    # contains models of getting properites, for example US standard atmosphere model
    name: Literal["air"] = pd.Field("air")
