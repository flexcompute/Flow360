"""This module allows users to write serializable, evaluable symbolic code for use in simulation params"""

from __future__ import annotations

import ast
import re
from numbers import Number
from typing import Annotated, Any, Generic, Iterable, Literal, Optional, TypeVar, Union

import pydantic as pd
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from pydantic_core import InitErrorDetails, core_schema
from typing_extensions import Self
from unyt import Unit, unyt_array

from flow360.component.simulation.blueprint import Evaluable, expr_to_model
from flow360.component.simulation.blueprint.core import EvaluationContext, expr_to_code
from flow360.component.simulation.blueprint.core.types import TargetSyntax
from flow360.component.simulation.blueprint.flow360.symbols import resolver
from flow360.component.simulation.framework.base_model import Flow360BaseModel

_global_ctx: EvaluationContext = EvaluationContext(resolver)
_user_variables: set[str] = set()
_solver_variable_name_map: dict[str, str] = {}


def _is_number_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _split_keep_delimiters(value: str, delimiters: list) -> list:
    escaped_delimiters = [re.escape(d) for d in delimiters]
    pattern = f"({'|'.join(escaped_delimiters)})"
    result = re.split(pattern, value)
    return [part for part in result if part != ""]


def _convert_numeric(value):
    arg = None
    unit_delimiters = ["+", "-", "*", "/", "(", ")"]
    if isinstance(value, Number):
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
        arg = f"{_convert_argument(value.value.tolist())[0]} * "
        for token in tokens:
            if token not in unit_delimiters and not _is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(value, list):
        arg = f"[{','.join([_convert_argument(item)[0] for item in value])}]"
    return arg


def _convert_argument(value):
    parenthesize = False
    arg = _convert_numeric(value)
    if isinstance(value, Expression):
        arg = value.expression
        parenthesize = True
    elif isinstance(value, Variable):
        arg = value.name
    if not arg:
        raise ValueError(f"Incompatible argument of type {type(value)}")
    return arg, parenthesize


class SerializedValueOrExpression(Flow360BaseModel):
    """Serialized frontend-compatible format of an arbitrary value/expression field"""

    type_name: Union[Literal["number"], Literal["expression"]] = pd.Field(None)
    value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None)
    units: Optional[str] = pd.Field(None)
    expression: Optional[str] = pd.Field(None)
    evaluated_value: Optional[Union[Number, Iterable[Number]]] = pd.Field(None)
    evaluated_units: Optional[str] = pd.Field(None)


# This is a wrapper to allow using unyt arrays with pydantic models
class UnytArray(unyt_array):
    """UnytArray wrapper to enable pydantic compatibility"""

    def __repr__(self):
        return f"UnytArray({str(self)})"

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any):
        """Minimal validator for pydantic compatibility"""
        if isinstance(value, unyt_array):
            return value
        raise ValueError(f"Cannot convert {type(value)} to UnytArray")


AnyNumericType = Union[float, UnytArray, list]


class Variable(Flow360BaseModel):
    """Base class representing a symbolic variable"""

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
        # pylint:disable=invalid-str-returned
        return self.name

    def __repr__(self):
        return f"Variable({self.name} = {self.value})"

    def __hash__(self):
        return hash(self.name)


class UserVariable(Variable):
    """Class representing a user-defined symbolic variable"""

    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        """Auto updating context when new variable is declared"""
        _global_ctx.set(value.name, value.value)
        _user_variables.add(value.name)
        return value

    @pd.model_validator(mode="after")
    @classmethod
    def check_dependencies(cls, value):
        """Validator for ensuring no cyclic dependency."""
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
    """Class representing a pre-defined symbolic variable that cannot be evaluated at client runtime"""

    solver_name: Optional[str] = pd.Field(None)

    @pd.model_validator(mode="after")
    @classmethod
    def update_context(cls, value):
        """Auto updating context when new variable is declared"""
        _global_ctx.set(value.name, value.value)
        _solver_variable_name_map[value.name] = (
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
    """
    A symbolic, validated representation of a mathematical expression.

    This model wraps a string-based expression, ensures its syntax and semantics
    against the global evaluation context, and provides methods to:
      - evaluate its numeric/unyt result (`evaluate`)
      - list user-defined variables it references (`user_variables` / `user_variable_names`)
      - emit C++ solver code (`to_solver_code`)
    """

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
        elif isinstance(value, list):
            expression = f"[{','.join([_convert_argument(item)[0] for item in value])}]"
        else:
            details = InitErrorDetails(
                type="value_error", ctx={"error": f"Invalid type {type(value)}"}
            )
            raise pd.ValidationError.from_exception_data("Expression type error", [details])
        try:
            expr_to_model(expression, _global_ctx)
        except SyntaxError as s_err:
            _handle_syntax_error(s_err, expression)
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("Expression value error", [details])

        return {"expression": expression}

    def evaluate(
        self, context: EvaluationContext = None, strict: bool = True
    ) -> Union[float, list, unyt_array]:
        """Evaluate this expression against the given context."""
        if context is None:
            context = _global_ctx
        expr = expr_to_model(self.expression, context)
        result = expr.evaluate(context, strict)
        return result

    def user_variables(self):
        """Get list of user variables used in expression."""
        expr = expr_to_model(self.expression, _global_ctx)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return [UserVariable(name=name, value=_global_ctx.get(name)) for name in names]

    def user_variable_names(self):
        """Get list of user variable names used in expression."""
        expr = expr_to_model(self.expression, _global_ctx)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return names

    def to_solver_code(self, params):
        """Convert to solver readable code."""

        def translate_symbol(name):
            if name in _solver_variable_name_map:
                return _solver_variable_name_map[name]

            if name in _user_variables:
                value = _global_ctx.get(name)
                if isinstance(value, Expression):
                    return f"{value.to_solver_code(params)}"
                return _convert_numeric(value)

            match = re.fullmatch("u\\.(.+)", name)

            if match:
                unit_name = match.group(1)
                unit = Unit(unit_name)
                conversion_factor = params.convert_unit(1.0 * unit, "flow360").v
                return str(conversion_factor)

            return name

        expr = expr_to_model(self.expression, _global_ctx)
        source = expr_to_code(expr, TargetSyntax.CPP, translate_symbol)
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
        # pylint:disable=invalid-str-returned
        return self.expression

    def __repr__(self):
        return f"Expression({self.expression})"

    def as_vector(self):
        """Parse the expression (str) and if possible, return list of `Expression` instances"""
        tree = ast.parse(self.expression, mode="eval")
        if isinstance(tree.body, ast.List):
            # Expression string with list syntax, like "[aa,bb,cc]"
            result = [ast.unparse(elt) for elt in tree.body.elts]
        else:
            # Expression string with **evaluated result**
            # being vector,like "[1,2,3]*u.m", "fl.cross(aa,bb)"

            # TODO: This seems to be a deadlock, here we depend on
            # string expression being properly (symbolically) evaluated.
            # Hacking so at least "[1,2,3]*u.m" works
            result = self.evaluate()
            print(">>> result = ", result, type(result))
            if isinstance(self, (list, unyt_array)):
                return result
            return None

        return [Expression.model_validate(item) for item in result]


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):
    """Model accepting both value and expressions"""

    def __class_getitem__(cls, typevar_values):  # pylint:disable=too-many-statements
        def _internal_validator(value: Expression):
            try:
                result = value.evaluate(strict=False)
            except Exception as err:
                raise ValueError(f"expression evaluation failed: {err}") from err
            pd.TypeAdapter(typevar_values).validate_python(result)
            return value

        expr_type = Annotated[Expression, pd.AfterValidator(_internal_validator)]

        def _deserialize(value) -> Self:
            def _validation_attempt_(input_value):
                deserialized = None
                try:
                    deserialized = SerializedValueOrExpression.model_validate(input_value)
                except:  # pylint:disable=bare-except
                    pass
                return deserialized

            ###
            deserialized = None
            if isinstance(value, dict) and "type_name" not in value:
                # Deserializing legacy simulation.json where there is only "units" + "value"
                deserialized = _validation_attempt_({**value, "type_name": "number"})
            else:
                deserialized = _validation_attempt_(value)
            if deserialized is None:
                # All validation attempt failed
                deserialized = value
            else:
                if deserialized.type_name == "number":
                    if deserialized.units is not None:
                        # Note: Flow360 unyt_array could not be constructed here.
                        return unyt_array(deserialized.value, deserialized.units)
                    return deserialized.value
                if deserialized.type_name == "expression":
                    return expr_type(expression=deserialized.expression)

            return deserialized

        def _serializer(value, info) -> dict:
            print(">>> Inside serializer: value = ", value)
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

        def _get_discriminator_value(v: Any) -> str:
            # Note: This is ran after deserializer
            if isinstance(v, SerializedValueOrExpression):
                return v.type_name
            if isinstance(v, dict):
                return v.get("typeName") if v.get("typeName") else v.get("type_name")
            if isinstance(v, (Expression, Variable, str)):
                return "expression"
            if isinstance(v, (Number, unyt_array, list)):
                return "number"
            raise KeyError("Unknown expression input type: ", v, v.__class__.__name__)

        union_type = Annotated[
            Union[
                Annotated[expr_type, Tag("expression")], Annotated[typevar_values, Tag("number")]
            ],
            Discriminator(_get_discriminator_value),
            BeforeValidator(_deserialize),
            PlainSerializer(_serializer),
        ]
        return union_type
