from pydantic_core import InitErrorDetails

from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.blueprint import expression_to_model

import pydantic as pd
from numbers import Number

_global_ctx: EvaluationContext = EvaluationContext()


def _convert_argument(other):
    parenthesize = False
    if isinstance(other, Flow360Expression):
        arg = other.body
        parenthesize = True
    elif isinstance(other, Flow360Variable):
        arg = other.name
    elif isinstance(other, Number):
        arg = str(other)
    else:
        raise ValueError(f"Incompatible argument of type {type(other)}")
    return arg, parenthesize


class Flow360Variable:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        _global_ctx.set(name, value)
    
    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.name} ** {str_arg}")

    def __neg__(self):
        return Flow360Expression(body=f"-{self.name}")

    def __pos__(self):
        return Flow360Expression(body=f"+{self.name}")

    def __abs__(self):
        return Flow360Expression(body=f"abs({self.name})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} + {self.name}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} - {self.name}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} * {self.name}")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} / {self.name}")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} // {self.name}")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} % {self.body}")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} ** {self.body}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Flow360Variable({self.name} = {self.value})"


class Flow360Expression(Flow360BaseModel):
    body: str = pd.Field("")

    @classmethod
    @pd.field_validator("body", mode="after")
    def _validate_expression(cls, value):
        try:
            _ = expression_to_model(value)
        except SyntaxError as s_err:
            details = InitErrorDetails(type="value_error", ctx={"error": s_err})
            raise pd.ValidationError.from_exception_data("expression syntax error", [details])
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("expression value error", [details])

        return value

    def evaluate(self) -> float:
        expr = expression_to_model(self.body)
        result = expr.evaluate(_global_ctx)
        return result

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.body} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{self.body} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"({self.body}) * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"({self.body}) / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"({self.body}) // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"({self.body}) % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"({self.body}) ** {str_arg}")

    def __neg__(self):
        return Flow360Expression(body=f"-({self.body})")

    def __pos__(self):
        return Flow360Expression(body=f"+({self.body})")

    def __abs__(self):
        return Flow360Expression(body=f"abs({self.body})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} + {self.body}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} - {self.body}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} * ({self.body})")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} / ({self.body})")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} // ({self.body})")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} % ({self.body})")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Flow360Expression(body=f"{str_arg} ** ({self.body})")

    def __str__(self):
        return self.body

    def __repr__(self):
        return f"Flow360Expression({self.body})"
