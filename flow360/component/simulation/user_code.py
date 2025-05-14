from __future__ import annotations

import re
from typing import ClassVar, Generic, Iterable, Optional, TypeVar

from pydantic import BeforeValidator
from typing_extensions import Self
from unyt import Unit, unyt_array

from flow360.component.simulation.blueprint import Evaluable, expr_to_model
from flow360.component.simulation.blueprint.codegen import expr_to_code
from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.blueprint.flow360 import resolver
from flow360.component.simulation.blueprint.utils.types import TargetSyntax
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import *

_global_ctx: EvaluationContext = EvaluationContext(resolver)
_user_variables: set[str] = set()
_solver_variables: dict[str, str] = dict()


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


def _convert_argument(value):
    parenthesize = False
    unit_delimiters = ["+", "-", "*", "/", "(", ")"]
    if isinstance(value, Expression):
        arg = value.expression
        parenthesize = True
    elif isinstance(value, Variable):
        arg = value.name
    elif isinstance(value, Number):
        arg = str(value)
    elif isinstance(value, Unit):
        unit = str(value)
        tokens = _split_keep_delimiters(unit, unit_delimiters)
        arg = ""
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(value, unyt_array):
        unit = str(value.units)
        tokens = _split_keep_delimiters(unit, unit_delimiters)
        arg = f"{_convert_argument(value.value)[0]} * "
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            arg = str(value)
        else:
            arg = f"np.array([{','.join([_convert_argument(item)[0] for item in value])}])"
    else:
        raise ValueError(f"Incompatible argument of type {type(value)}")
    return arg, parenthesize


class SerializedValueOrExpression(Flow360BaseModel):
    type_name: Union[Literal["number"], Literal["expression"]] = pd.Field(None)
    value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None)
    units: Optional[str] = pd.Field(None)
    expression: Optional[str] = pd.Field(None)
    evaluated_value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None)
    evaluated_units: Optional[str] = pd.Field(None)


# This is a wrapper to allow using ndarrays with pydantic models
class NdArray(np.ndarray):
    def __repr__(self):
        return f"NdArray(shape={self.shape}, dtype={self.dtype})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any):
        if isinstance(value, np.ndarray):
            return value
        raise ValueError(f"Cannot convert {type(value)} to NdArray")


# This is a wrapper to allow using unyt arrays with pydantic models
class UnytArray(unyt_array):
    def __repr__(self):
        return f"UnytArray({str(self)})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any):
        if isinstance(value, unyt_array):
            return value
        raise ValueError(f"Cannot convert {type(value)} to UnytArray")


AnyNumericType = Union[float, UnytArray, NdArray]


class Variable(Flow360BaseModel):
    name: str = pd.Field()
    value: ValueOrExpression[AnyNumericType] = pd.Field()

    model_config = pd.ConfigDict(validate_assignment=True, extra="allow")

    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} + {str_arg}")

    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} - {str_arg}")

    def __mul__(self, other):
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

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
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

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

    def __getitem__(self, item):
        (arg, _) = _convert_argument(item)
        return Expression(expression=f"{self.name}[{arg}]")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Variable({self.name} = {self.value})"

    def __hash__(self):
        return hash(self.name)

    def sqrt(self):
        return Expression(expression=f"np.sqrt({self.expression})")

    def sin(self):
        return Expression(expression=f"np.sin({self.expression})")

    def cos(self):
        return Expression(expression=f"np.cos({self.expression})")

    def tan(self):
        return Expression(expression=f"np.tan({self.expression})")

    def arcsin(self):
        return Expression(expression=f"np.arcsin({self.expression})")

    def arccos(self):
        return Expression(expression=f"np.arccos({self.expression})")

    def arctan(self):
        return Expression(expression=f"np.arctan({self.expression})")


class UserVariable(Variable):
    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        _global_ctx.set(value.name, value.value)
        _user_variables.add(value.name)
        return value

    @pd.model_validator(mode="after")
    @classmethod
    def check_dependencies(cls, value):
        visited = set()
        stack = [(value.name, [value.name])]
        while stack:
            (current_name, current_path) = stack.pop()
            current_value = _global_ctx.get(current_name)
            if isinstance(current_value, Expression):
                used_names = current_value.user_variable_names()
                if [name for name in used_names if name in current_path]:
                    path_string = " -> ".join(current_path + [current_path[0]])
                    details = InitErrorDetails(
                        type="value_error",
                        ctx={"error": f"Cyclic dependency between variables {path_string}"},
                    )
                    raise pd.ValidationError.from_exception_data("Variable value error", [details])
                stack.extend(
                    [(name, current_path + [name]) for name in used_names if name not in visited]
                )
        return value


class SolverVariable(Variable):
    solver_name: Optional[str] = pd.Field(None)

    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        _global_ctx.set(value.name, value.value)
        _solver_variables[value.name] = (
            value.solver_name if value.solver_name is not None else value.name
        )
        return value


def _handle_syntax_error(se: SyntaxError, source: str):
    caret = " " * (se.offset - 1) + "^" if se.text and se.offset else None
    msg = f"{se.msg} at line {se.lineno}, column {se.offset}"
    if caret:
        msg += f"\n{se.text.rstrip()}\n{caret}"

    raise pd.ValidationError.from_exception_data(
        "expression_syntax",
        [
            InitErrorDetails(
                type="value_error",
                msg=se.msg,
                input=source,
                ctx={
                    "line": se.lineno,
                    "column": se.offset,
                    "error": msg,
                },
            )
        ],
    )


class Expression(Flow360BaseModel, Evaluable):
    expression: str = pd.Field("")

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="before")
    @classmethod
    def _validate_expression(cls, value) -> Self:
        if isinstance(value, str):
            expression = value
        elif isinstance(value, dict) and "expression" in value.keys():
            expression = value["expression"]
        elif isinstance(value, Expression):
            expression = str(value)
        elif isinstance(value, Variable):
            expression = str(value)
        elif isinstance(value, np.ndarray) and not isinstance(value, unyt_array):
            if value.ndim == 0:
                expression = str(value)
            else:
                expression = (
                    f"np.array([{','.join([_convert_argument(item)[0] for item in value])}])"
                )
        else:
            details = InitErrorDetails(
                type="value_error", ctx={"error": f"Invalid type {type(value)}"}
            )
            raise pd.ValidationError.from_exception_data("Expression type error", [details])
        try:
            expr_to_model(expression, _global_ctx)
        except SyntaxError as s_err:
            raise _handle_syntax_error(s_err, expression)
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("Expression value error", [details])

        return {"expression": expression}

    def evaluate(
        self, context: EvaluationContext = None, strict: bool = True
    ) -> Union[float, np.ndarray, unyt_array]:
        if context is None:
            context = _global_ctx
        expr = expr_to_model(self.expression, context)
        result = expr.evaluate(context, strict)
        return result

    def user_variables(self):
        expr = expr_to_model(self.expression, _global_ctx)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return [UserVariable(name=name, value=_global_ctx.get(name)) for name in names]

    def user_variable_names(self):
        expr = expr_to_model(self.expression, _global_ctx)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return names

    def to_solver_code(self):
        expr = expr_to_model(self.expression, _global_ctx)
        source = expr_to_code(expr, TargetSyntax.CPP, _solver_variables, "u.*")
        return source

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
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

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
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

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

    def __getitem__(self, item):
        (arg, _) = _convert_argument(item)
        return Expression(expression=f"({self.expression})[{arg}]")

    def __str__(self):
        return self.expression

    def __repr__(self):
        return f"Expression({self.expression})"

    def sqrt(self):
        return Expression(expression=f"np.sqrt({self.expression})")

    def sin(self):
        return Expression(expression=f"np.sin({self.expression})")

    def cos(self):
        return Expression(expression=f"np.cos({self.expression})")

    def tan(self):
        return Expression(expression=f"np.tan({self.expression})")

    def arcsin(self):
        return Expression(expression=f"np.arcsin({self.expression})")

    def arccos(self):
        return Expression(expression=f"np.arccos({self.expression})")

    def arctan(self):
        return Expression(expression=f"np.arctan({self.expression})")


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):
    def __class_getitem__(cls, internal_type):
        def _internal_validator(value: Expression):
            try:
                result = value.evaluate(strict=False)
            except Exception as err:
                raise ValueError(f"expression evaluation failed: {err}") from err
            pd.TypeAdapter(internal_type).validate_python(result)
            return value

        expr_type = Annotated[Expression, pd.AfterValidator(_internal_validator)]

        def _deserialize(value) -> Self:
            is_serialized = False

            try:
                value = SerializedValueOrExpression.model_validate(value)
                is_serialized = True
            except Exception as err:
                pass

            if is_serialized:
                if value.type_name == "number":
                    if value.units is not None:
                        return unyt_array(value.value, value.units)
                    else:
                        return value.value
                elif value.type_name == "expression":
                    return expr_type(expression=value.expression)
            else:
                return value

        def _serializer(value, info) -> dict:
            if isinstance(value, Expression):
                serialized = SerializedValueOrExpression(type_name="expression")

                serialized.expression = value.expression

                evaluated = value.evaluate(strict=False)

                if isinstance(evaluated, Number):
                    serialized.evaluated_value = evaluated
                elif isinstance(evaluated, unyt_array):

                    if evaluated.size == 1:
                        serialized.evaluated_value = float(evaluated.value)
                    else:
                        serialized.evaluated_value = tuple(evaluated.value.tolist())

                    serialized.evaluated_units = str(evaluated.units.expr)
            else:
                serialized = SerializedValueOrExpression(type_name="number")
                if isinstance(value, Number):
                    serialized.value = value
                elif isinstance(value, unyt_array):

                    if value.size == 1:
                        serialized.value = float(value.value)
                    else:
                        serialized.value = tuple(value.value.tolist())

                    serialized.units = str(value.units.expr)

            return serialized.model_dump(**info.__dict__)

        union_type = Union[expr_type, internal_type]

        union_type = Annotated[union_type, PlainSerializer(_serializer)]
        union_type = Annotated[union_type, BeforeValidator(_deserialize)]

        return union_type
