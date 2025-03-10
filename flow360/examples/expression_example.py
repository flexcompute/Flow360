from typing import Union

from rich import print

import pydantic as pd

from flow360.component.simulation.expressions import Flow360Expression, Flow360Variable
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360 import u


class ExampleModel(Flow360BaseModel):
    foo: Union[float, Flow360Expression] = pd.Field(None)


x = Flow360Variable(name="x", value=4 * u.m / u.s)

bar_expr = ExampleModel(foo=x * u.m / u.s ** 2)

print(str(bar_expr.foo))
print(bar_expr.foo.evaluate())
