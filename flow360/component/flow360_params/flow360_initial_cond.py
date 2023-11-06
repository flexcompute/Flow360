import pydantic as pd

from .params_base import (
    Flow360BaseModel
)


class InitialCondition(Flow360BaseModel):
    type: str


class FreestreamInitialCondition(InitialCondition):
    type = pd.Field("Freestream", const=True)


class ExpressionInitialCondition(InitialCondition):
    type = pd.Field("Expression", const=True)
    rho: str = pd.Field()
    u: str = pd.Field()
    v: str = pd.Field()
    w: str = pd.Field()
    p: str = pd.Field()

