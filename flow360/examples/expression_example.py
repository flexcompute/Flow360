from typing import Union

from rich import print

import pydantic as pd

from flow360.component.simulation.expressions import ExpressionField
from flow360.component.simulation.framework.base_model import Flow360BaseModel


class ExampleModel(Flow360BaseModel):
    foo: Union[float, ExpressionField] = pd.Field(None)


bar_value = ExampleModel(foo=22)

bar_expr = ExampleModel(foo="2 + 4 * 5")

print(bar_value.foo)
print(bar_expr.foo.evaluate())
