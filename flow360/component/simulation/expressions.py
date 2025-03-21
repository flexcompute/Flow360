from typing import get_origin, Generic, TypeVar, Self, Optional
import re

from flow360.component.simulation.blueprint.flow360 import resolver
from flow360.component.simulation.unit_system import *
from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.blueprint import expression_to_model

import pydantic as pd
from numbers import Number
from unyt import Unit, unyt_quantity, unyt_array


_global_ctx: EvaluationContext = EvaluationContext(resolver)


def _is_descendant_of(t, base):
    if t is None:
        return False
    origin = get_origin(t) or t
    return issubclass(origin, base)


def _is_number_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _split_keep_delimiters(input: str, delimiters: list) -> list:
    escaped_delimiters = [re.escape(d) for d in delimiters]
    pattern = f"({'|'.join(escaped_delimiters)})"
    result = re.split(pattern, input)
    return [part for part in result if part != ""]


def _convert_argument(other):
    parenthesize = False
    unit_delimiters = ["+", "-", "*", "/", "(", ")"]
    if isinstance(other, Expression):
        arg = other.body
        parenthesize = True
    elif isinstance(other, Variable):
        arg = other.name
    elif isinstance(other, Number):
        arg = str(other)
    elif isinstance(other, Unit):
        unit = str(other)
        tokens = _split_keep_delimiters(unit, unit_delimiters)
        arg = ""
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(other, unyt_quantity):
        unit = str(other.units)
        tokens = _split_keep_delimiters(unit, unit_delimiters)
        arg = f"{str(other.value)} * "
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(other, unyt_array):
        unit = str(other.units)
        tokens = _split_keep_delimiters(unit, unit_delimiters)
        arg = f"{str(other.value)} * "
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    else:
        raise ValueError(f"Incompatible argument of type {type(other)}")
    return arg, parenthesize


class Variable(Flow360BaseModel):
    name: str = pd.Field()
    value: Optional[Union[list[float], float, unyt_quantity, unyt_array]] = pd.Field(float('NaN'))

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        _global_ctx.set(value.name, value.value)

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.name} ** {str_arg}")

    def __neg__(self):
        return Expression(body=f"-{self.name}")

    def __pos__(self):
        return Expression(body=f"+{self.name}")

    def __abs__(self):
        return Expression(body=f"abs({self.name})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} + {self.name}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} - {self.name}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} * {self.name}")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} / {self.name}")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} // {self.name}")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} % {self.name}")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} ** {self.name}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Flow360Variable({self.name} = {self.value})"


def _get_internal_validator(internal_type):
    def _internal_validator(value: Expression):
        result = value.evaluate()
        pd.TypeAdapter(internal_type).validate_python(result)
        return value

    return _internal_validator


class Expression(Flow360BaseModel):
    body: str = pd.Field("")

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="wrap")
    @classmethod
    def _validate_expression(cls, value, handler) -> Self:
        if isinstance(value, str):
            body = value
        elif isinstance(value, dict) and "body" in value.keys():
            body = value["body"]
        elif isinstance(value, Expression):
            body = value.body
        elif isinstance(value, Variable):
            body = str(value)
        else:
            details = InitErrorDetails(
                type="value_error", ctx={"error": f"Invalid type {type(value)}"}
            )
            raise pd.ValidationError.from_exception_data("expression type error", [details])

        try:
            _ = expression_to_model(body, _global_ctx)
        except SyntaxError as s_err:
            details = InitErrorDetails(type="value_error", ctx={"error": s_err})
            raise pd.ValidationError.from_exception_data("expression syntax error", [details])
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("expression value error", [details])

        return handler({"body": body})

    def evaluate(self) -> float:
        expr = expression_to_model(self.body, _global_ctx)
        result = expr.evaluate(_global_ctx)
        return result

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.body} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{self.body} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"({self.body}) * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"({self.body}) / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"({self.body}) // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"({self.body}) % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"({self.body}) ** {str_arg}")

    def __neg__(self):
        return Expression(body=f"-({self.body})")

    def __pos__(self):
        return Expression(body=f"+({self.body})")

    def __abs__(self):
        return Expression(body=f"abs({self.body})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} + {self.body}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} - {self.body}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} * ({self.body})")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} / ({self.body})")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} // ({self.body})")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} % ({self.body})")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(body=f"{str_arg} ** ({self.body})")

    def __str__(self):
        return self.body

    def __repr__(self):
        return f"Flow360Expression({self.body})"


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):

    def __class_getitem__(cls, internal_type):
        if isinstance(internal_type, Number):

            def _non_dimensional_validator(value):
                result = value.evaluate()
                if isinstance(result, Number):
                    return value
                msg = "The evaluated value needs to be a non-dimensional scalar"
                details = InitErrorDetails(type="value_error", ctx={"error": msg})
                raise pd.ValidationError.from_exception_data("expression value error", [details])

            expr_type = Annotated[Expression, pd.AfterValidator(_non_dimensional_validator)]
        else:
            expr_type = Annotated[
                Expression, pd.AfterValidator(_get_internal_validator(internal_type))
            ]

        return Union[internal_type, expr_type]

    @pd.model_validator(mode="wrap")
    @classmethod
    def _convert_to_dict(cls, value, handler) -> Self:
        value = Expression.model_validate(value)
        return handler({"body": value.body})
