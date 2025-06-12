"""This module allows users to write serializable, evaluable symbolic code for use in simulation params"""

from __future__ import annotations

import ast
import re
from numbers import Number
from typing import Annotated, Any, Generic, List, Literal, Optional, TypeVar, Union

import numpy as np
import pydantic as pd
import unyt as u
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from pydantic_core import InitErrorDetails, core_schema
from typing_extensions import Self
from unyt import Unit, unyt_array, unyt_quantity

from flow360.component.simulation.blueprint import Evaluable, expr_to_model
from flow360.component.simulation.blueprint.core import EvaluationContext, expr_to_code
from flow360.component.simulation.blueprint.core.types import TargetSyntax
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.user_code.core.context import default_context
from flow360.component.simulation.user_code.core.utils import (
    handle_syntax_error,
    is_number_string,
    split_keep_delimiters,
)

_user_variables: set[str] = set()
_solver_variables: set[str] = set()


def __soft_fail_add__(self, other):
    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__add__(self, other)
    return NotImplemented


def __soft_fail_sub__(self, other):
    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__sub__(self, other)
    return NotImplemented


def __soft_fail_mul__(self, other):
    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__mul__(self, other)
    return NotImplemented


def __soft_fail_truediv__(self, other):
    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__truediv__(self, other)
    return NotImplemented


unyt_array.__add__ = __soft_fail_add__
unyt_array.__sub__ = __soft_fail_sub__
unyt_array.__mul__ = __soft_fail_mul__
unyt_array.__truediv__ = __soft_fail_truediv__


def _convert_numeric(value):
    arg = None
    unit_delimiters = ["+", "-", "*", "/", "(", ")"]
    if isinstance(value, Number):
        arg = str(value)
    elif isinstance(value, Unit):
        unit = str(value)
        tokens = split_keep_delimiters(unit, unit_delimiters)
        arg = ""
        for token in tokens:
            if token not in unit_delimiters and not is_number_string(token):
                token = f"u.{token}"
                arg += token
            else:
                arg += token
    elif isinstance(value, unyt_array):
        unit = str(value.units)
        tokens = split_keep_delimiters(unit, unit_delimiters)
        arg = f"{_convert_argument(value.value.tolist())[0]} * "
        for token in tokens:
            if token not in unit_delimiters and not is_number_string(token):
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

    type_name: Literal["number", "expression"] = pd.Field()
    value: Optional[Union[Number, list[Number]]] = pd.Field(None)
    units: Optional[str] = pd.Field(None)
    expression: Optional[str] = pd.Field(None)
    evaluated_value: Union[Optional[Number], list[Optional[Number]]] = pd.Field(None)
    evaluated_units: Optional[str] = pd.Field(None)
    output_units: Optional[str] = pd.Field(None, description="See definition in `Expression`.")


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

    def __eq__(self, other):
        # NaN-compatible equal operator for unit test support
        if not isinstance(other, Variable):
            return False
        return self.model_dump_json() == other.model_dump_json()


class UserVariable(Variable):
    """Class representing a user-defined symbolic variable"""

    @pd.model_validator(mode="after")
    def update_context(self):
        """Auto updating context when new variable is declared"""
        default_context.set(self.name, self.value)
        _user_variables.add(self.name)
        return self

    @pd.model_validator(mode="after")
    def check_dependencies(self):
        """Validator for ensuring no cyclic dependency."""
        visited = set()
        stack = [(self.name, [self.name])]
        while stack:
            (current_name, current_path) = stack.pop()
            current_value = default_context.get(current_name)
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
        return self

    def __hash__(self):
        """
        Support for set and deduplicate.
        Can be removed if not used directly in output_fields.
        """
        return hash(self.model_dump_json())

    def in_unit(self, new_unit: Union[str, Unit] = None):
        """Requesting the output of the variable to be in the given (new_unit) units."""
        if isinstance(new_unit, Unit):
            new_unit = str(new_unit)
        self.value.output_units = new_unit
        return self


class SolverVariable(Variable):
    """Class representing a pre-defined symbolic variable that cannot be evaluated at client runtime"""

    solver_name: Optional[str] = pd.Field(None)
    variable_type: Literal["Volume", "Surface", "Scalar"] = pd.Field()

    @pd.model_validator(mode="after")
    def update_context(self):
        """Auto updating context when new variable is declared"""
        default_context.set(self.name, self.value, Variable)
        _solver_variables.add(self.name)
        if self.solver_name:
            default_context.set_alias(self.name, self.solver_name)
        return self

    def in_unit(self, new_name: str, new_unit: Union[str, Unit] = None):
        """


        Return a UserVariable that will generate results in the new_unit.
        If new_unit is not specified then the unit will be determined by the unit system.
        """
        if isinstance(new_unit, Unit):
            new_unit = str(new_unit)
        new_variable = UserVariable(name=new_name, value=Expression(expression=self.name))
        new_variable.value.output_units = new_unit  # pylint:disable=assigning-non-slot
        return new_variable


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
    output_units: Optional[str] = pd.Field(
        None,
        description="String representation of what the requested units the evaluated expression should be "
        "when `self` is used as an output field. By default the output units will be inferred from the unit "
        "system associated with SimulationParams",
    )

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="before")
    @classmethod
    def _validate_expression(cls, value) -> Self:
        output_units = None
        if isinstance(value, str):
            expression = value
        elif isinstance(value, dict) and "expression" in value.keys():
            expression = value["expression"]
            output_units = value.get("output_units")
        elif isinstance(value, Expression):
            expression = str(value)
            output_units = value.output_units
        elif isinstance(value, Variable):
            expression = str(value)
            if isinstance(value.value, Expression):
                output_units = value.value.output_units
        elif isinstance(value, list):
            expression = f"[{','.join([_convert_argument(item)[0] for item in value])}]"
        else:
            details = InitErrorDetails(
                type="value_error", ctx={"error": f"Invalid type {type(value)}"}
            )
            raise pd.ValidationError.from_exception_data("Expression type error", [details])
        try:
            # To ensure the expression is valid (also checks for
            expr_to_model(expression, default_context)
            # To reduce unnecessary parentheses
            expression = ast.unparse(ast.parse(expression))
        except SyntaxError as s_err:
            handle_syntax_error(s_err, expression)
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("Expression value error", [details])

        return {"expression": expression, "output_units": output_units}

    def evaluate(
        self,
        context: EvaluationContext = None,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Union[float, list[float], unyt_array, Expression]:
        """Evaluate this expression against the given context."""
        if context is None:
            context = default_context
        expr = expr_to_model(self.expression, context)
        result = expr.evaluate(context, raise_on_non_evaluable, force_evaluate)

        # Sometimes we may yield a list of expressions instead of
        # an expression containing a list, so we check this here
        # and convert if necessary

        if isinstance(result, list):
            is_expression_list = False

            for item in result:
                if isinstance(item, Expression):
                    is_expression_list = True

            if is_expression_list:
                result = Expression.model_validate(result)

        return result

    def user_variables(self):
        """Get list of user variables used in expression."""
        expr = expr_to_model(self.expression, default_context)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return [UserVariable(name=name, value=default_context.get(name)) for name in names]

    def user_variable_names(self):
        """Get list of user variable names used in expression."""
        expr = expr_to_model(self.expression, default_context)
        names = expr.used_names()
        names = [name for name in names if name in _user_variables]

        return names

    def solver_variable_names(self):
        """Get list of solver variable names used in expression."""
        expr = expr_to_model(self.expression, default_context)
        names = expr.used_names()
        names = [name for name in names if name in _solver_variables]
        return names

    def to_solver_code(self, params):
        """Convert to solver readable code."""

        def translate_symbol(name):
            alias = default_context.get_alias(name)

            if alias:
                return alias

            match = re.fullmatch("u\\.(.+)", name)

            if match:
                unit_name = match.group(1)
                unit = Unit(unit_name)
                conversion_factor = params.convert_unit(1.0 * unit, "flow360").v
                return str(conversion_factor)

            return name

        partial_result = self.evaluate(
            default_context, raise_on_non_evaluable=False, force_evaluate=False
        )

        if isinstance(partial_result, Expression):
            expr = expr_to_model(partial_result.expression, default_context)
        else:
            expr = expr_to_model(_convert_numeric(partial_result), default_context)

        return expr_to_code(expr, TargetSyntax.CPP, translate_symbol)

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

    def __getitem__(self, index):
        (arg, _) = _convert_argument(index)
        tree = ast.parse(self.expression, mode="eval")
        int_arg = None
        try:
            int_arg = int(arg)
        except ValueError:
            pass
        if isinstance(tree.body, ast.List) and int_arg is not None:
            # Expression string with list syntax, like "[aa,bb,cc]"
            # and since the index is static we can reduce it
            result = [ast.unparse(elt) for elt in tree.body.elts]
            return Expression.model_validate(result[int_arg])
        return Expression(expression=f"({self.expression})[{arg}]")

    def __str__(self):
        # pylint:disable=invalid-str-returned
        return self.expression

    def __repr__(self):
        return f"Expression({self.expression})"

    @property
    def dimensionality(self):
        """The physical dimensionality of the expression."""
        value = self.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
        assert isinstance(value, (unyt_array, unyt_quantity))
        return value.units.dimensions

    @property
    def length(self):
        """The number of elements in the expression."""
        value = self.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
        assert isinstance(
            value, (unyt_array, unyt_quantity, list, Number)
        ), f"Unexpected evaluated result type: {type(value)}"
        if isinstance(value, list):
            return len(value)
        return 1 if isinstance(value, (unyt_quantity, Number)) else value.shape[0]

    def get_output_units(self, input_params=None):
        """
        Get the output units of the expression.

        - If self.output_units is None, derive the default output unit based on the
        value's dimensionality and current unit system.

        - If self.output_units is valid u.Unit string, deserialize it and return it.

        - If self.output_units is valid unit system name, derive the default output
        unit based on the value's dimensionality and the **given** unit system.

        - If expression is a number constant, return None.

        - Else raise ValueError.
        """

        def get_unit_from_unit_system(expression: Expression, unit_system_name: str):
            """Derive the default output unit based on the value's dimensionality and current unit system"""
            numerical_value = expression.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            if not isinstance(numerical_value, (u.unyt_array, u.unyt_quantity, list)):
                # Pure dimensionless constant
                return None
            if isinstance(numerical_value, list):
                numerical_value = numerical_value[0]

            if unit_system_name in ("SI", "SI_unit_system"):
                return numerical_value.in_base("mks").units
            if unit_system_name in ("Imperial", "Imperial_unit_system"):
                return numerical_value.in_base("imperial").units
            if unit_system_name in ("CGS", "CGS_unit_system"):
                return numerical_value.in_base("cgs").units
            raise ValueError(f"[Internal] Invalid unit system: {unit_system_name}")

        try:
            return u.Unit(self.output_units)
        except u.exceptions.UnitParseError as e:
            if input_params is None:
                raise ValueError(
                    "[Internal] input_params required when output_units is not valid u.Unit string"
                ) from e
            if not self.output_units:
                unit_system_name: Literal["SI", "Imperial", "CGS"] = input_params.unit_system.name
            else:
                unit_system_name = self.output_units
            return get_unit_from_unit_system(self, unit_system_name)


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):
    """Model accepting both value and expressions"""

    def __class_getitem__(cls, typevar_values):  # pylint:disable=too-many-statements
        def _internal_validator(value: Expression):
            try:
                result = value.evaluate(raise_on_non_evaluable=False)
            except Exception as err:
                raise ValueError(f"expression evaluation failed: {err}") from err
            pd.TypeAdapter(typevar_values).validate_python(result, context={"allow_inf_nan": True})
            return value

        expr_type = Annotated[Expression, pd.AfterValidator(_internal_validator)]

        def _deserialize(value) -> Self:
            try:
                value = SerializedValueOrExpression.model_validate(value)
                if value.type_name == "number":
                    if value.units is not None:
                        # unyt objects
                        return unyt_array(value.value, value.units)
                    return value.value
                if value.type_name == "expression":
                    return expr_type(expression=value.expression, output_units=value.output_units)
            except Exception:  # pylint:disable=broad-exception-caught
                pass

            return value

        def _serializer(value, info) -> dict:
            if isinstance(value, Expression):
                serialized = SerializedValueOrExpression(
                    type_name="expression",
                    output_units=value.output_units,
                )

                serialized.expression = value.expression

                evaluated = value.evaluate(raise_on_non_evaluable=False)

                if isinstance(evaluated, Number):
                    serialized.evaluated_value = evaluated
                elif isinstance(evaluated, unyt_array):
                    if evaluated.size == 1:
                        serialized.evaluated_value = (
                            float(evaluated.value)
                            if not np.isnan(evaluated.value)
                            else None  # NaN-None handling
                        )
                    else:
                        serialized.evaluated_value = tuple(
                            item if not np.isnan(item) else None
                            for item in evaluated.value.tolist()
                        )

                    serialized.evaluated_units = str(evaluated.units.expr)
            else:
                serialized = SerializedValueOrExpression(type_name="number")
                if isinstance(value, (Number, List)):
                    serialized.value = value
                elif isinstance(value, unyt_array):
                    if value.size == 1:
                        serialized.value = float(value.value)
                    else:
                        serialized.value = tuple(value.value.tolist())

                    serialized.units = str(value.units.expr)

            return serialized.model_dump(**info.__dict__)

        def _discriminator(v: Any) -> str:
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
            pd.Field(discriminator=Discriminator(_discriminator)),
            BeforeValidator(_deserialize),
            PlainSerializer(_serializer),
        ]
        return union_type
