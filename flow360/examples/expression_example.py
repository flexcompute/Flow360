from typing import Union

from rich import print

import pydantic as pd

from flow360.component.simulation.expressions import Flow360Expression, Flow360Variable
from flow360.component.simulation.framework.base_model import Flow360BaseModel


class ExampleModel(Flow360BaseModel):
    foo: Union[float, Flow360Expression] = pd.Field(None)


x = Flow360Variable(name="x", value=4)
y = Flow360Variable(name="y", value=2)

bar_expr = ExampleModel(foo=x + 2 - (x ** 2) / y)

print(str(bar_expr.foo))
print(bar_expr.foo.evaluate())
