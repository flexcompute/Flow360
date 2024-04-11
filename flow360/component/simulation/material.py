import pydantic as pd

from flow360.component.flow360_params.params_base import Flow360BaseModel


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...


class Material(Flow360BaseModel):
    # contains models of getting properites, for example US standard atmosphere model
    name: Literal["air"] = pd.Field("air")
