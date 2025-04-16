from __future__ import annotations
from typing import get_origin, Generic, TypeVar, Optional, Iterable

from pydantic import WrapSerializer, WrapValidator
from typing_extensions import Self
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
_user_variables: set[str] = set()


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
        arg = other.expression
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


class SerializedValueOrExpression(Flow360BaseModel):
    type_name: Union[Literal["number"], Literal["expression"]] = pd.Field(None, alias="typeName")
    value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None)
    units: Optional[str] = pd.Field(None)
    expression: Optional[str] = pd.Field(None)
    evaluated_value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None, alias="evaluatedValue")
    evaluated_units: Optional[str] = pd.Field(None, alias="evaluatedUnits")


class Variable(Flow360BaseModel):
    name: str = pd.Field()
    value: Union[list[float], float, unyt_quantity, unyt_array] = pd.Field()

    model_config = pd.ConfigDict(validate_assignment=True)

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} ** {str_arg}")

    def __neg__(self):
        return Expression(expression=f"-{self.name}")

    def __pos__(self):
        return Expression(expression=f"+{self.name}")

    def __abs__(self):
        return Expression(expression=f"abs({self.name})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.name}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.name}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * {self.name}")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / {self.name}")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // {self.name}")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % {self.name}")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} ** {self.name}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Variable({self.name} = {self.value})"


class UserVariable(Variable):
    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        _global_ctx.set(value.name, value.value)
        _user_variables.add(value.name)


class SolverVariable(Variable):
    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        _global_ctx.set(value.name, value.value)


class Expression(Flow360BaseModel):
    expression: str = pd.Field("")

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="wrap")
    @classmethod
    def _validate_expression(cls, value, handler) -> Self:
        if isinstance(value, str):
            expression = value
        elif isinstance(value, dict) and "expression" in value.keys():
            expression = value["expression"]
        elif isinstance(value, Expression):
            expression = value.expression
        elif isinstance(value, Variable):
            expression = str(value)
        else:
            details = InitErrorDetails(
                type="value_error", ctx={"error": f"Invalid type {type(value)} for {value}"}
            )
            raise pd.ValidationError.from_exception_data("expression type error", [details])

        try:
            _ = expression_to_model(expression, _global_ctx)
        except SyntaxError as s_err:
            details = InitErrorDetails(type="value_error", ctx={"error": s_err})
            raise pd.ValidationError.from_exception_data("expression syntax error", [details])
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("expression value error", [details])

        return handler({"expression": expression})

    def evaluate(self, strict=True) -> float:
        expr = expression_to_model(self.expression, _global_ctx)
        result = expr.evaluate(_global_ctx, strict)
        return result

    def user_variables(self):
        expr = expression_to_model(self.expression, _global_ctx)
        names = expr.used_names()

        names = [name for name in names if name in _user_variables]

        return [UserVariable(name=name, value=_global_ctx.get(name)) for name in names]

    def __hash__(self):
        return hash(self.expression)

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.expression} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.expression} - {str_arg}")

    def __mul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) * {str_arg}")

    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) / {str_arg}")

    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) // {str_arg}")

    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) ** {str_arg}")

    def __neg__(self):
        return Expression(expression=f"-({self.expression})")

    def __pos__(self):
        return Expression(expression=f"+({self.expression})")

    def __abs__(self):
        return Expression(expression=f"abs({self.expression})")

    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.expression}")

    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.expression}")

    def __rmul__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * ({self.expression})")

    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / ({self.expression})")

    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // ({self.expression})")

    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % ({self.expression})")

    def __rpow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} ** ({self.expression})")

    def __str__(self):
        return self.expression

    def __repr__(self):
        return f"Expression({self.expression})"


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):
    def __class_getitem__(cls, internal_type):
        def _internal_validator(value: Expression):
            result = value.evaluate(strict=False)
            validated = pd.TypeAdapter(internal_type).validate_python(result, strict=True)
            return value

        expr_type = Annotated[Expression, pd.AfterValidator(_internal_validator)]

        def _deserialize(value, handler) -> Self:
            try:
                value = SerializedValueOrExpression.model_validate(value, strict=True)
                if value.type_name == "number":
                    if value.units is not None:
                        return handler(unyt_quantity(value.value, value.units))
                    else:
                        return handler(value.value)
                elif value.type_name == "expression":
                    return handler(value.expression)
            except Exception as err:
                pass

            return handler(value)

        def _serializer(value, handler, info) -> dict:
            if isinstance(value, Expression):
                serialized = SerializedValueOrExpression(typeName="expression")

                serialized.expression = value.expression

                evaluated = value.evaluate(strict=False)

                if isinstance(evaluated, Number):
                    serialized.evaluated_value = evaluated
                elif isinstance(evaluated, unyt_quantity) or isinstance(evaluated, unyt_array):

                    if evaluated.size == 1:
                        serialized.evaluated_value = float(evaluated.value)
                    else:
                        serialized.evaluated_value = tuple(evaluated.value.tolist())

                    serialized.evaluated_units = str(evaluated.units.expr)
            else:
                serialized = SerializedValueOrExpression(typeName="number")
                if isinstance(value, Number):
                    serialized.value = value
                elif isinstance(value, unyt_quantity) or isinstance(value, unyt_array):

                    if value.size == 1:
                        serialized.value = float(value.value)
                    else:
                        serialized.value = tuple(value.value.tolist())

                    serialized.units = str(value.units.expr)

            return serialized.model_dump(**info.__dict__)

        return Annotated[Annotated[Union[expr_type, internal_type], WrapSerializer(_serializer)], WrapValidator(_deserialize)]
