# pylint: disable=too-many-lines
"""This module allows users to write serializable, evaluable symbolic code for use in simulation params"""

from __future__ import annotations

import ast
import re
from numbers import Number
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import pydantic as pd
import unyt as u
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from pydantic_core import InitErrorDetails, core_schema
from typing_extensions import Self
from unyt import Unit, dimensions, unyt_array, unyt_quantity

from flow360.component.simulation.blueprint import Evaluable, expr_to_model
from flow360.component.simulation.blueprint.core import EvaluationContext, expr_to_code
from flow360.component.simulation.blueprint.core.types import TargetSyntax
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater_utils import deprecation_reminder
from flow360.component.simulation.unit_system import unit_system_manager
from flow360.component.simulation.user_code.core.context import default_context
from flow360.component.simulation.user_code.core.utils import (
    handle_syntax_error,
    is_number_string,
    split_keep_delimiters,
)

_solver_variables: set[str] = set()


class VariableContextInfo(Flow360BaseModel):
    """Variable context info for project variables."""

    name: str
    value: ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore
    postProcessing: bool = pd.Field()

    # pylint: disable=fixme
    # TODO: This should be removed once front end figure out what to store here.
    model_config = pd.ConfigDict(extra="allow")


def save_user_variables(params):
    """
    Save user variables to the project variables.
    Declared here since I do not want to import default_context everywhere.
    """
    # Get all output variables:
    post_processing_variables = set()
    for item in params.outputs if params.outputs else []:
        if not "output_fields" in item.__class__.model_fields:
            continue
        for item in item.output_fields.items:
            if isinstance(item, UserVariable):
                post_processing_variables.add(item.name)

    params.private_attribute_asset_cache.project_variables = [
        VariableContextInfo(
            name=name, value=value, postProcessing=name in post_processing_variables
        )
        for name, value in default_context._values.items()  # pylint: disable=protected-access
        if "." not in name  # Skipping scoped variables (non-user variables)
    ]
    return params


def update_global_context(value: List[VariableContextInfo]):
    """Once the project variables are validated, update the global context."""

    for item in value:
        default_context.set(item.name, item.value)
    return value


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


def __soft_fail_pow__(self, other):
    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__pow__(self, other)  # pylint: disable=too-many-function-args
    return NotImplemented


unyt_array.__add__ = __soft_fail_add__
unyt_array.__sub__ = __soft_fail_sub__
unyt_array.__mul__ = __soft_fail_mul__
unyt_array.__truediv__ = __soft_fail_truediv__
unyt_array.__pow__ = __soft_fail_pow__


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


class UnytQuantity(unyt_quantity):
    """UnytQuantity wrapper to enable pydantic compatibility"""

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any):
        """Minimal validator for pydantic compatibility"""
        if isinstance(value, unyt_quantity):
            return value
        if isinstance(value, unyt_array):
            # When deserialized unyt_quantity() gives unyt_array
            if value.shape == ():
                return unyt_quantity(value.value, value.units)
        raise ValueError("Input should be a valid unit quantity.")


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


def check_vector_arithmetic(func):
    """Decorator to check if vector arithmetic is being attempted and raise an error if so."""

    def wrapper(self, other):
        def is_array(item):
            if isinstance(item, unyt_array) and item.shape != ():
                return True
            if isinstance(item, list):
                return True
            return False

        if is_array(self.value) or is_array(other):
            raise ValueError(
                f"Vector operation ({func.__name__} between {self.name} and {other}) not "
                "supported for variables. Please write expression for each component."
            )
        return func(self, other)

    return wrapper


class Variable(Flow360BaseModel):
    """Base class representing a symbolic variable"""

    name: str = pd.Field(frozen=True)

    model_config = pd.ConfigDict(validate_assignment=True)

    @property
    def value(self):
        """
        Get the value of the variable from the global context.
        """
        return default_context.get(self.name)

    @value.setter
    def value(self, value):
        """
        Set the value of the variable in the global context.
        In parallel to `set_value` this supports syntax like `my_user_var.value = 10.0`.
        """
        new_value = pd.TypeAdapter(
            ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]
        ).validate_python(value)
        # Not checking overwrite here since it is user controlled explicit assignment operation
        default_context.set(self.name, new_value)

    @pd.model_validator(mode="before")
    @classmethod
    def set_value(cls, values):
        """
        Supporting syntax like `a = fl.Variable(name="a", value=1)`.
        """
        if "name" not in values:
            raise ValueError("`name` is required for variable declaration.")

        if "value" in values:
            new_value = pd.TypeAdapter(
                ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]
            ).validate_python(values.pop("value"))
            # Check overwriting, skip for solver variables:
            if values["name"] in default_context.user_variable_names:
                diff = new_value != default_context.get(values["name"])

                if isinstance(diff, np.ndarray):
                    diff = diff.any()

                if isinstance(diff, list):
                    # Might not end up here but just in case
                    diff = any(diff)

                if diff:
                    raise ValueError(
                        f"Redeclaring user variable '{values['name']}' with new value: {new_value}. "
                        f"Previous value: {default_context.get(values['name'])}"
                    )
            # Call the setter
            default_context.set(
                values["name"],
                new_value,
            )

        return values

    @check_vector_arithmetic
    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} + {str_arg}")

    @check_vector_arithmetic
    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} - {str_arg}")

    @check_vector_arithmetic
    def __mul__(self, other):
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} * {str_arg}")

    @check_vector_arithmetic
    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} / {str_arg}")

    @check_vector_arithmetic
    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} // {str_arg}")

    @check_vector_arithmetic
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

    @check_vector_arithmetic
    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.name}")

    @check_vector_arithmetic
    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.name}")

    @check_vector_arithmetic
    def __rmul__(self, other):
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * {self.name}")

    @check_vector_arithmetic
    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / {self.name}")

    @check_vector_arithmetic
    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // {self.name}")

    @check_vector_arithmetic
    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % {self.name}")

    @check_vector_arithmetic
    def __rpow__(self, other):
        (arg, _) = _convert_argument(other)
        str_arg = f"({arg})"  # Always parenthesize to ensure base is evaluated first
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

    def __len__(self):
        """The number of elements in self.value. 0 for scalar and anything else for vector."""
        if isinstance(self.value, Expression):
            return len(self.value)
        if isinstance(self.value, unyt_array):
            # Can be either unyt_array or unyt_quantity
            if self.value.shape == ():
                return 0
            # No 2D arrays are supported
            return self.value.shape[0]
        if isinstance(self.value, list):
            return len(self.value)
        if isinstance(self.value, Number):
            return 0
        raise ValueError(f"Cannot get length information for {self.value}")


class UserVariable(Variable):
    """Class representing a user-defined symbolic variable"""

    name: str = pd.Field(frozen=True)
    type_name: Literal["UserVariable"] = pd.Field("UserVariable", frozen=True)

    @pd.field_validator("name", mode="after")
    @classmethod
    @deprecation_reminder("25.7.0")
    def check_value_is_not_legacy_variable(cls, v):
        """Check that the value is not a legacy variable"""
        # pylint:disable=import-outside-toplevel
        from flow360.component.simulation.outputs.output_fields import AllFieldNames

        all_field_names = set(AllFieldNames.__args__)
        if v in all_field_names:
            raise ValueError(f"'{v}' is a reserved (legacy) output field name.")
        return v

    @pd.field_validator("name", mode="after")
    @classmethod
    def check_valid_user_variable_name(cls, v):
        """Validate a variable identifier (ASCII only)."""
        # Partial list of keywords; extend as needed
        RESERVED_SYNTAX_KEYWORDS = {  # pylint:disable=invalid-name
            "int",
            "double",
            "float",
            "long",
            "short",
            "char",
            "bool",
            "void",
            "class",
            "for",
            "while",
            "if",
            "else",
            "return",
            "namespace",
            "template",
            "typename",
            "constexpr",
            "virtual",
        }
        if not v:
            raise ValueError("Identifier cannot be empty.")

        # 2) First character must be letter or underscore
        if not re.match(r"^[A-Za-z_]", v):
            raise ValueError("Identifier must start with a letter (A-Z/a-z) or underscore (_).")

        # 3) All characters must be letters, digits, or underscore
        if re.search(r"[^A-Za-z0-9_]", v):
            raise ValueError(
                "Identifier can only contain letters, digits (0-9), or underscore (_)."
            )

        # 4) Not a C++ keyword
        if v in RESERVED_SYNTAX_KEYWORDS:
            raise ValueError(f"'{v}' is a reserved keyword.")

        # 5) existing variable name:
        solver_side_names = {
            item.split(".")[-1] for item in default_context.registered_names if "." in item
        }
        if v in solver_side_names:
            raise ValueError(f"'{v}' is a reserved solver side variable name.")

        return v

    def __hash__(self):
        """
        Support for set and deduplicate.
        """
        return hash(self.model_dump_json())

    def in_units(
        self,
        new_unit: Union[
            str, Literal["SI_unit_system", "CGS_unit_system", "Imperial_unit_system"], Unit
        ] = None,
    ):
        """Requesting the output of the variable to be in the given (new_unit) units."""
        if isinstance(new_unit, Unit):
            new_unit = str(new_unit)
        if not isinstance(self.value, Expression):
            raise ValueError("Cannot set output units for non expression value.")
        self.value.output_units = new_unit
        return self


class SolverVariable(Variable):
    """Class representing a pre-defined symbolic variable that cannot be evaluated at client runtime"""

    type_name: Literal["SolverVariable"] = pd.Field("SolverVariable", frozen=True)
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

    def in_units(
        self,
        new_name: str,
        new_unit: Union[
            str, Literal["SI_unit_system", "CGS_unit_system", "Imperial_unit_system"], Unit
        ] = None,
    ):
        """
        Return a UserVariable that will generate results in the new_unit.
        If new_unit is not specified then the unit will be determined by the unit system.
        """
        if isinstance(new_unit, Unit):
            new_unit = str(new_unit)
        new_variable = UserVariable(
            name=new_name,
            value=Expression(expression=self.name),
        )
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

    @pd.field_validator("expression", mode="after")
    @classmethod
    def remove_leading_and_trailing_whitespace(cls, value: str) -> str:
        """Remove leading and trailing whitespace from the expression"""
        return value.strip()

    @pd.field_validator("expression", mode="after")
    @classmethod
    def disable_relative_temperature_scale(cls, value: str) -> str:
        """Disable relative temperature scale usage"""
        if "u.degF" in value or "u.degC" in value:
            raise ValueError(
                "Relative temperature scale usage is not allowed. Please use u.R or u.K instead."
            )
        return value

    @pd.model_validator(mode="after")
    def check_output_units_matches_dimensions(self) -> str:
        """Check that the output units have the same dimensions as the expression"""
        if not self.output_units:
            return self
        if self.output_units in ("SI_unit_system", "CGS_unit_system", "Imperial_unit_system"):
            return self
        output_units_dimensions = u.Unit(self.output_units).dimensions
        expression_dimensions = self.dimensions
        if output_units_dimensions != expression_dimensions:
            raise ValueError(
                f"Output units '{self.output_units}' have different dimensions "
                f"{output_units_dimensions} than the expression {expression_dimensions}."
            )

        return self

    @pd.field_validator("output_units", mode="after")
    @classmethod
    def disable_relative_temperature_scale_in_output_units(cls, value: str) -> str:
        """Disable relative temperature scale usage in output units"""
        if not value:
            return value
        if "u.degF" in value or "u.degC" in value:
            raise ValueError(
                "Relative temperature scale usage is not allowed in output units. Please use u.R or u.K instead."
            )
        return value

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
        names = [name for name in names if name in default_context.user_variable_names]

        return [UserVariable(name=name, value=default_context.get(name)) for name in names]

    def user_variable_names(self):
        """Get list of user variable names used in expression."""
        expr = expr_to_model(self.expression, default_context)
        names = expr.used_names()
        names = [name for name in names if name in default_context.user_variable_names]

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

            # solver-time resolvable functions:
            func_match = re.fullmatch(r"math\.(.+)", name)
            if func_match:
                func_name = func_match.group(1)
                return func_name

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
        (arg, _) = _convert_argument(other)
        str_arg = f"({arg})"  # Always parenthesize to ensure base is evaluated first
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

    def __eq__(self, other):
        if isinstance(other, Expression):
            return self.expression == other.expression
        return super().__eq__(other)

    @property
    def dimensions(self):
        """The physical dimensions of the expression."""
        value = self.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
        assert isinstance(
            value, (unyt_array, unyt_quantity, list, Number)
        ), "Non unyt array so no dimensions"
        if isinstance(value, (unyt_array, unyt_quantity)):
            return value.units.dimensions
        if isinstance(value, list):
            _check_list_items_are_same_dimensions(value)
            return value[0].units.dimensions
        return u.Unit("dimensionless").dimensions

    @property
    def length(self):
        """The number of elements in the expression. 0 for scalar and anything else for vector."""
        return get_input_value_length(self)

    def __len__(self):
        return self.length

    def get_output_units(self, input_params=None):
        """
        Get the output units of the expression.

        - If self.output_units is None, derive the default output unit based on the
        value's dimensions and current unit system.

        - If self.output_units is valid u.Unit string, deserialize it and return it.

        - If self.output_units is valid unit system name, derive the default output
        unit based on the value's dimensions and the **given** unit system.

        - If expression is a number constant, return None.

        - Else raise ValueError.
        """

        def get_unit_from_unit_system(expression: Expression, unit_system_name: str):
            """Derive the default output unit based on the value's dimensions and current unit system"""
            numerical_value = expression.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            if isinstance(numerical_value, list):
                numerical_value = numerical_value[0]
            if not isinstance(numerical_value, (u.unyt_array, u.unyt_quantity)):
                # Pure dimensionless constant
                return None
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
            # The unit system for inferring the units for input has different temperature unit
            u.unit_systems.imperial_unit_system["temperature"] = u.Unit("R").expr
            result = get_unit_from_unit_system(self, unit_system_name)
            u.unit_systems.imperial_unit_system["temperature"] = u.Unit("degF").expr
            return result


def _check_list_items_are_same_dimensions(value: list):
    if all(isinstance(item, Expression) for item in value):
        _check_list_items_are_same_dimensions(
            [item.evaluate(raise_on_non_evaluable=False, force_evaluate=True) for item in value]
        )
        return
    if all(isinstance(item, unyt_quantity) for item in value):
        # ensure all items have the same dimensions
        if not all(item.units.dimensions == value[0].units.dimensions for item in value):
            raise ValueError("All items in the list must have the same dimensions.")
        return
    # Also raise when some elements is Number and others are unyt_quantity
    if any(isinstance(item, Number) for item in value) and any(
        isinstance(item, unyt_quantity) for item in value
    ):
        raise ValueError("List must contain only all unyt_quantities or all numbers.")
    return


T = TypeVar("T")


class ValueOrExpression(Expression, Generic[T]):
    """Model accepting both value and expressions"""

    _cfg: ClassVar[dict] = {}

    @classmethod
    def configure(cls, **flags):
        """
        Create a new subclass with the given flags.
        """
        name = f"{cls.__name__}[{','.join(f'{k}={v}' for k,v in flags.items())}]"
        return type(name, (cls,), {"_cfg": {**cls._cfg, **flags}})

    def __class_getitem__(cls, typevar_values):  # pylint:disable=too-many-statements
        cfg = cls._cfg
        # By default all value or expression should be able to be evaluated at compile-time
        allow_run_time_expression = bool(cfg.get("allow_run_time_expression", False))

        def _internal_validator(value: Expression):
            try:
                # Symbolicly validate
                value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
                # Numerically validate
                result = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            except Exception as err:
                raise ValueError(f"expression evaluation failed: {err}") from err

            # Detect run-time expressions
            if allow_run_time_expression is False:
                if is_runtime_expression(result):
                    raise ValueError(
                        "Run-time expression is not allowed in this field. "
                        "Please ensure this field does not depend on any control or solver variables."
                    )

            pd.TypeAdapter(typevar_values).validate_python(
                result, context={"allow_inf_nan": allow_run_time_expression}
            )
            return value

        expr_type = Annotated[Expression, pd.AfterValidator(_internal_validator)]

        def _deserialize(value) -> Self:
            # Try to see if the value is already a SerializedValueOrExpression
            try:
                value = SerializedValueOrExpression.model_validate(value)
            except Exception:  # pylint:disable=broad-exception-caught
                pass
            if isinstance(value, SerializedValueOrExpression):
                if value.type_name == "number":
                    if value.units is not None:
                        # unyt objects
                        return unyt_array(value.value, value.units, dtype=np.float64)
                    return value.value
                if value.type_name == "expression":
                    return expr_type(expression=value.expression, output_units=value.output_units)

            @deprecation_reminder("25.8.0")
            def _handle_legacy_unyt_values(value):
                """Handle {"units":..., "value":...} from legacy input. This is much easier than writing the updater."""
                if isinstance(value, dict) and "units" in value and "value" in value:
                    return unyt_array(value["value"], value["units"], dtype=np.float64), True
                return value, False

            value, is_legacy_unyt_value = _handle_legacy_unyt_values(value)
            if is_legacy_unyt_value:
                return value

            # Handle list of unyt_quantities:
            if isinstance(value, list):
                # Only checking when list[unyt_quantity]
                if len(value) == 0:
                    raise ValueError("Empty list is not allowed.")
                _check_list_items_are_same_dimensions(value)
                if all(isinstance(item, (unyt_quantity, Number)) for item in value):
                    # try limiting the number of types we need to handle
                    return unyt_array(value, dtype=np.float64)
            return value

        def _serializer(value, info) -> dict:
            if isinstance(value, Expression):
                serialized = SerializedValueOrExpression(
                    type_name="expression",
                    output_units=value.output_units,
                )

                serialized.expression = value.expression

                evaluated = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)

                if isinstance(evaluated, list):
                    # May result from Expression which is actually a list of expressions
                    try:
                        evaluated = u.unyt_array(evaluated, dtype=np.float64)
                    except u.exceptions.IterableUnitCoercionError:
                        # Inconsistent units for components of list
                        pass

                if isinstance(evaluated, Number):
                    serialized.evaluated_value = (
                        evaluated if not np.isnan(evaluated) else None  # NaN-None handling
                    )
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
                # Note: NaN handling should be unnecessary since it would
                # have end up being expression first so not reaching here.
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
            if isinstance(v, list) and all(isinstance(item, Expression) for item in v):
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


def is_runtime_expression(value):
    """Check if the input value is a runtime expression."""
    if isinstance(value, unyt_quantity) and np.isnan(value.value):
        return True
    if isinstance(value, unyt_array) and np.isnan(value.value).any():
        return True
    if isinstance(value, Number) and np.isnan(value):
        return True
    if isinstance(value, list) and any(np.isnan(item) for item in value):
        return any(np.isnan(item) for item in value)
    return False


def get_input_value_dimensions(
    value: Union[float, list[float], unyt_array, unyt_quantity, Expression, Variable],
):
    """Get the dimensions of the input value."""
    if isinstance(value, list):
        return get_input_value_dimensions(value=value[0]) if len(value) > 0 else None
    if isinstance(value, Variable):
        return get_input_value_dimensions(value=value.value)
    if isinstance(value, Expression):
        return value.dimensions
    if isinstance(value, (unyt_array, unyt_quantity)):
        return value.units.dimensions
    if isinstance(value, Number):
        return dimensions.dimensionless
    raise ValueError(
        "Cannot get input value's dimensions due to the unknown value type: ",
        value,
        value.__class__.__name__,
    )


def get_input_value_length(
    value: Union[Number, list[float], unyt_array, unyt_quantity, Expression, Variable],
):
    """Get the length of the input value."""
    if isinstance(value, Expression):
        value = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
    assert isinstance(
        value, (unyt_array, unyt_quantity, list, Number, np.ndarray)
    ), f"Unexpected evaluated result type: {type(value)}"
    if isinstance(value, list):
        return len(value)
    if isinstance(value, np.ndarray):
        return 0 if value.shape == () else value.shape[0]
    return 0 if isinstance(value, (unyt_quantity, Number)) else value.shape[0]


def solver_variable_to_user_variable(item):
    """Convert the solver variable to a user variable using the current unit system."""
    if isinstance(item, SolverVariable):
        if unit_system_manager.current is None:
            raise ValueError(f"Solver variable {item.name} cannot be used without a unit system.")
        unit_system_name = unit_system_manager.current.name
        name = item.name.split(".")[-1] if "." in item.name else item.name
        return UserVariable(name=f"{name}_{unit_system_name}", value=item)
    return item
