from rich import print

import pydantic as pd

from flow360.component.simulation.expressions import Flow360ValueOrExpression, Flow360Variable
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360 import u
from flow360.component.simulation.unit_system import LengthType


class ExampleModel(Flow360BaseModel):
    field_1: Flow360ValueOrExpression[float] = pd.Field(None)
    field_2: Flow360ValueOrExpression[LengthType.NonNegative] = pd.Field(None)
    field_3: Flow360ValueOrExpression[LengthType] = pd.Field(None)
    field_4: Flow360ValueOrExpression[LengthType.Vector] = pd.Field(None)


x = Flow360Variable(name="x", value=4)

y = Flow360Variable(name="y", value=[0, 0, 1])

bar_1 = ExampleModel(
    field_1=4.0,
    field_2=x * u.m + 6.0 * u.m,
    field_3=x * u.m - 6.0 * u.m,
    field_4=y * u.m
)

print(str(bar_1.field_1))
print(str(bar_1.field_2))
print(str(bar_1.field_3))
print(str(bar_1.field_4))

print(bar_1.field_1)
print(bar_1.field_2.evaluate())
print(bar_1.field_3.evaluate())
print(bar_1.field_4.evaluate())