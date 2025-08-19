# pylint: disable=too-many-lines
"""This module allows users to write serializable, evaluable symbolic code for use in simulation params"""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from enum import Enum
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
    SOLVER_INTERNAL_VARIABLES,
    handle_syntax_error,
    is_number_string,
    is_runtime_expression,
    split_keep_delimiters,
)
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.log import log

_solver_variables: dict[str, str] = {}


class VariableContextInfo(Flow360BaseModel):
    """Variable context info for project variables."""

    name: str
    value: ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore
    description: Optional[str] = pd.Field(None)

    @pd.field_validator("value", mode="after")
    @classmethod
    def convert_number_to_expression(cls, value: AnyNumericType) -> ValueOrExpression:
        """So that frontend can properly display the value of the variable."""
        if not isinstance(value, Expression):
            return Expression.model_validate(_convert_numeric(value))
        return value


def update_global_context(value: List[VariableContextInfo]):
    """Once the project variables are validated, update the global context."""

    for item in value:
        default_context.set(item.name, item.value)
    return value


def get_user_variable(name: str):
    """Get the user variable from the global context."""
    return UserVariable(name=name, value=default_context.get(name))


def remove_user_variable(name: str):
    """Remove the variable from the global context."""
    return default_context.remove(name)


def show_user_variables():
    """Show the user variables from the global context with name and value in two columns, wrapping long values."""
    # pylint: disable=too-many-locals
    user_variables = {
        name: default_context.get(name) for name in sorted(default_context.user_variable_names)
    }

    if not user_variables.keys():
        log.info("No user variables are currently defined.")
        return

    header_index = "Idx"
    header_name = "Name"
    header_value = "Value"

    max_name_width = max(max(len(name) for name in user_variables.keys()), len(header_name))

    terminal_width = 100

    value_col_width = max(terminal_width - (len(header_index) + 1 + max_name_width), 20)

    formatted_header = (
        f"{header_index:>{len(header_index)}}. "
        f"{header_name:<{max_name_width}} "
        f"{header_value}"
    )
    separator = f"{'-'*(len(header_index)+1)} " f"{'-'*max_name_width} " f"{'-'*value_col_width}"

    output_lines = [formatted_header, separator]

    for idx, (name, value) in enumerate(user_variables.items()):
        value_str = str(value)

        value_lines_raw = value_str.splitlines()

        wrapped_value_lines = []
        for line in value_lines_raw:
            wrapped_line_parts = textwrap.wrap(line, width=value_col_width)
            wrapped_value_lines.extend(wrapped_line_parts)

        first_value_line = wrapped_value_lines[0] if wrapped_value_lines else ""

        output_lines.append(
            f"{idx+1:>{len(header_index)}}. {name:<{max_name_width}} {first_value_line}"
        )

        indent_for_wrapped_lines = " " * (len(header_index) + max_name_width + 2)

        for subsequent_line in wrapped_value_lines[1:]:
            output_lines.append(f"{indent_for_wrapped_lines}{subsequent_line}")

    output_lines = "\n".join(output_lines)

    log.info(f"The current defined user variables are:\n{output_lines}")


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


def _is_array(item):
    if isinstance(item, unyt_array) and item.shape != ():
        return True
    if isinstance(item, list):
        return True
    return False


def check_vector_binary_arithmetic(func):
    """Decorator to check if vector arithmetic is being attempted and raise an error if so."""

    def wrapper(self, other):
        if _is_array(self.value) or _is_array(other):
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
        In parallel to `deserialize` this supports syntax like `my_user_var.value = 10.0`.
        """
        new_value = pd.TypeAdapter(
            ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]
        ).validate_python(value)
        # Not checking overwrite here since it is user controlled explicit assignment operation
        default_context.set(self.name, new_value)

    @pd.model_validator(mode="before")
    @classmethod
    def deserialize(cls, values):
        """
        Supporting syntax like `a = fl.Variable(name="a", value=1, description="some description")`.
        """
        if "name" not in values:
            raise ValueError("`name` is required for variable declaration.")

        if "value" in values:
            new_value = pd.TypeAdapter(
                ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]
            ).validate_python(values.pop("value"))

            # Check redeclaration, skip for solver variables:
            if values["name"] in default_context.user_variable_names:
                registered_expression = VariableContextInfo.convert_number_to_expression(
                    default_context.get(values["name"])
                )
                registered_expression_stripped = registered_expression.expression.replace(" ", "")

                if isinstance(new_value, Expression):
                    new_value_stripped = new_value.expression.replace(" ", "")
                else:
                    new_value_stripped = VariableContextInfo.convert_number_to_expression(
                        new_value
                    ).expression.replace(" ", "")

                if new_value_stripped != registered_expression_stripped:
                    raise ValueError(
                        f"Redeclaring user variable '{values['name']}' with new value: {new_value}. "
                        f"Previous value: {default_context.get(values['name'])}"
                    )
            else:
                # No conflict, call the setter
                default_context.set(
                    values["name"],
                    new_value,
                )

        if "description" in values and values["description"] is not None:
            if not isinstance(values["description"], str):
                raise ValueError(
                    f"Description must be a string but got {type(values['description'])}."
                )
            default_context.set_metadata(values["name"], "description", values["description"])
        values.pop("description", None)

        return values

    @check_vector_binary_arithmetic
    def __add__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} + {str_arg}")

    @check_vector_binary_arithmetic
    def __sub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} - {str_arg}")

    @check_vector_binary_arithmetic
    def __mul__(self, other):
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} * {str_arg}")

    @check_vector_binary_arithmetic
    def __truediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} / {str_arg}")

    @check_vector_binary_arithmetic
    def __floordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} // {str_arg}")

    @check_vector_binary_arithmetic
    def __mod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} % {str_arg}")

    def __pow__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        if _is_array(self.value):
            components = [f"{self.name}[{i}] ** {str_arg}" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")
        return Expression(expression=f"{self.name} ** {str_arg}")

    def __neg__(self):
        if _is_array(self.value):
            components = [f"-{self.name}[{i}]" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")
        return Expression(expression=f"-{self.name}")

    def __pos__(self):
        if _is_array(self.value):
            components = [f"+{self.name}[{i}]" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")
        return Expression(expression=f"+{self.name}")

    @check_vector_binary_arithmetic
    def __radd__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.name}")

    @check_vector_binary_arithmetic
    def __rsub__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.name}")

    @check_vector_binary_arithmetic
    def __rmul__(self, other):
        if isinstance(other, Number) and other == 0:
            return Expression(expression="0")

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * {self.name}")

    @check_vector_binary_arithmetic
    def __rtruediv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / {self.name}")

    @check_vector_binary_arithmetic
    def __rfloordiv__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // {self.name}")

    @check_vector_binary_arithmetic
    def __rmod__(self, other):
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % {self.name}")

    @check_vector_binary_arithmetic
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
    @deprecation_reminder("25.8.0")
    def check_value_is_not_legacy_variable(cls, v):
        """Check that the value is not a legacy variable"""
        # pylint:disable=import-outside-toplevel
        from flow360.component.simulation.outputs.output_fields import AllFieldNames

        all_field_names = set(AllFieldNames.__args__)
        if v in all_field_names:
            raise ValueError(
                f"'{v}' is a reserved (legacy) output field name. It cannot be used in expressions."
            )
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
        solver_side_names = solver_side_names.union(SOLVER_INTERNAL_VARIABLES)
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
        _solver_variables.update({self.name: self.variable_type})
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


_feature_requirement_map = {
    "solution.nu_hat": (
        lambda x: x.feature_usage.turbulence_model_type == "SpalartAllmaras",
        "Spalart-Allmaras turbulence solver is not used.",
    ),
    "solution.turbulence_kinetic_energy": (
        lambda x: x.feature_usage.turbulence_model_type == "kOmegaSST",
        "k-omega turbulence solver is not used.",
    ),
    "solution.specific_rate_of_dissipation": (
        lambda x: x.feature_usage.turbulence_model_type == "kOmegaSST",
        "k-omega turbulence solver is not used.",
    ),
    "solution.amplification_factor": (
        lambda x: x.feature_usage.transition_model_type == "AmplificationFactorTransport",
        "Amplification factor transition model is not used.",
    ),
    "solution.turbulence_intermittency": (
        lambda x: x.feature_usage.transition_model_type == "AmplificationFactorTransport",
        "Amplification factor transition model is not used.",
    ),
    "solution.density": (
        lambda x: x.using_liquid_as_material is False,
        "Liquid operating condition is used.",
    ),
    "solution.temperature": (
        lambda x: x.using_liquid_as_material is False,
        "Liquid operating condition is used.",
    ),
    "solution.Mach": (
        lambda x: x.using_liquid_as_material is False,
        "Liquid operating condition is used.",
    ),
    "control.physicalStep": (
        lambda x: x.time_stepping == "Unsteady",
        "Unsteady time stepping is not used.",
    ),
    "control.timeStepSize": (
        lambda x: x.time_stepping == "Unsteady",
        "Unsteady time stepping is not used.",
    ),
    "control.theta": (
        lambda x: x.feature_usage.rotation_zone_count == 0,
        "Rotation zone is not used.",
    ),
    "control.omega": (
        lambda x: x.feature_usage.rotation_zone_count == 0,
        "Rotation zone is not used.",
    ),
    "control.omegaDot": (
        lambda x: x.feature_usage.rotation_zone_count == 0,
        "Rotation zone is not used.",
    ),
}


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
    def sanitize_expression(cls, value: str) -> str:
        """Remove leading and trailing whitespace from the expression"""
        return value.strip().rstrip("; \n\t")

    @pd.field_validator("expression", mode="after")
    @classmethod
    def disable_confusing_operators(cls, value: str) -> str:
        """Disable confusing operators. This ideally should be warnings but we do not have warning system yet."""
        if "^" in value:
            raise ValueError(
                "^ operator is not allowed in expressions. For power operator, please use ** instead."
            )
        # This has no possible usage yet.
        if "&" in value:
            raise ValueError(
                "& operator is not allowed in expressions."  # . For logical AND use 'and' instead."
            )
        return value

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

    @pd.model_validator(mode="after")
    def ensure_dependent_feature_enabled(self) -> str:
        """
        Ensure that all dependent features are enabled for all the solver variables.
        Remaining checks:
        1. variable valid source check.
        2. variable location check.

        """
        validation_info = get_validation_info()
        if validation_info is None or self.expression not in validation_info.referenced_expressions:
            return self
        # Setting recursive to False to avoid recursive error message.
        # All user variables will be checked anyways.
        for solver_variable_name in self.solver_variable_names(recursive=False):
            if solver_variable_name in _feature_requirement_map:
                if not _feature_requirement_map[solver_variable_name][0](validation_info):
                    raise ValueError(
                        f"`{solver_variable_name}` cannot be used "
                        f"because {_feature_requirement_map[solver_variable_name][1]}"
                    )
        return self

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

    def solver_variable_names(
        self,
        recursive: bool,
        variable_type: Literal["Volume", "Surface", "Scalar", "All"] = "All",
    ):
        """Get list of solver variable names used in expression, recursively checking user variables.

        Params:
        -------
        - variable_type: The type of variable to get the names of.
        - recursive: Whether to recursively check user variables for solver variables.
        """

        def _get_solver_variable_names_recursive(
            expression: Expression, visited: set[str], recursive: bool
        ) -> set[str]:
            """Recursively get solver variable names from expression and its user variables."""
            solver_names = set()

            # Prevent infinite recursion by tracking visited expressions
            expr_str = str(expression)
            if expr_str in visited:
                return solver_names
            visited.add(expr_str)

            # Get solver variables directly from this expression
            expr = expr_to_model(expression.expression, default_context)
            names = expr.used_names()
            direct_solver_names = [name for name in names if name in _solver_variables]
            solver_names.update(direct_solver_names)

            if not recursive:
                return solver_names

            # Get user variables from this expression and recursively check their values
            user_vars = expression.user_variables()
            for user_var in user_vars:
                try:
                    if isinstance(user_var.value, Expression):
                        # Recursively check the user variable's expression
                        recursive_solver_names = _get_solver_variable_names_recursive(
                            user_var.value, visited, recursive
                        )
                        solver_names.update(recursive_solver_names)
                except (ValueError, AttributeError):
                    # Handle cases where user variable might not be properly defined
                    pass

            return solver_names

        # Start the recursive search
        all_solver_names = _get_solver_variable_names_recursive(self, set(), recursive)

        # Filter by variable type if specified
        if variable_type != "All":
            all_solver_names = {
                name for name in all_solver_names if _solver_variables[name] == variable_type
            }

        return list(all_solver_names)

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
                if unit == u.dimensionless:  # pylint:disable=no-member
                    return "1.0"
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
                return u.Unit("dimensionless")
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
                # Symbolically validate
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
            # Temporary suspend unit system to expose dimension problem
            unit_system_manager.suspend()
            pd.TypeAdapter(typevar_values).validate_python(
                result, context={"allow_inf_nan": allow_run_time_expression}
            )
            unit_system_manager.resume()
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
                    if value.expression is None:
                        raise ValueError("No expression found in the input")
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


def get_post_processing_variables(params) -> set[str]:
    """
    Get all the post processing related variables from the simulation params.
    """
    post_processing_variables = set()
    for item in params.outputs if params.outputs else []:
        if item.output_type in ("IsosurfaceOutput", "TimeAverageIsosurfaceOutput"):
            for isosurface in item.entities.items:
                if isinstance(isosurface.field, UserVariable):
                    post_processing_variables.add(isosurface.field.name)
        if not "output_fields" in item.__class__.model_fields:
            continue
        for item in item.output_fields.items:
            if isinstance(item, UserVariable):
                post_processing_variables.add(item.name)
    return post_processing_variables


def save_user_variables(params):
    """
    Save user variables to the project variables.
    Declared here since I do not want to import default_context everywhere.
    """
    # pylint:disable=protected-access
    for name, value in default_context._values.items():
        if "." in name:
            continue
        if params.private_attribute_asset_cache.variable_context is None:
            params.private_attribute_asset_cache.variable_context = []
        params.private_attribute_asset_cache.variable_context.append(
            VariableContextInfo(
                name=name,
                value=value,
                description=default_context.get_metadata(name, "description"),
            )
        )
    return params


def batch_get_user_variable_units(variable_names: list[str], params):
    """
    Get the units of a list of user variables.
    """
    result = {}
    for name in variable_names:
        value = default_context.get(name)
        if isinstance(value, Expression):
            result[name] = value.get_output_units(params)
        elif isinstance(value, unyt_array):
            result[name] = value.units
        elif isinstance(value, Number):
            result[name] = "dimensionless"
        else:
            raise ValueError(f"Unknown variable type: {value}")
    return result


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


def solver_variable_to_user_variable(item):
    """Convert the solver variable to a user variable using the current unit system."""
    if isinstance(item, SolverVariable):
        if unit_system_manager.current is None:
            raise ValueError(f"Solver variable {item.name} cannot be used without a unit system.")
        unit_system_name = unit_system_manager.current.name
        name = item.name.split(".")[-1] if "." in item.name else item.name
        return UserVariable(name=f"{name}_{unit_system_name}", value=item)
    return item


def get_referenced_expressions_and_user_variables(param_as_dict: dict):
    """
    Get all the expressions that are mentioned/referenced in the params dict
    (excluding the ones that are in the asset cache)
    Two sources:
    1. Field is `Expression`.
    2. Field is `UserVariable` and `value` is an `Expression`.
    `Expression` and `UserVariable` are both identified by their schema.
    """

    class ExpressionUsage(Enum):
        """
        Enum to identify the usage of an expression.
        """

        VALUE_OR_EXPRESSION = "ValueOrExpression"
        EXPRESSION_AS_IS = "ExpressionAsIs"

    def _is_user_variable(field: dict) -> bool:
        return "type_name" in field and field["type_name"] == "UserVariable"

    def _is_expression(field: dict) -> bool:
        # Two possible cases:
        # ValueOrExpression:
        if "type_name" in field and field["type_name"] == "expression":
            return ExpressionUsage.VALUE_OR_EXPRESSION
        # Expression as is (no such usage as of now)
        if sorted(field.keys()) == ["expression", "output_units"] or sorted(field.keys()) == [
            "expression"
        ]:
            return ExpressionUsage.EXPRESSION_AS_IS
        return False

    def _get_dependent_expressions(
        expression: Expression,
        dependent_expressions: set[str],
    ) -> list[str]:
        """
        Get all the expressions that are dependent on the given expression.
        """
        for var in expression.user_variables():
            try:
                if "." not in var.name and isinstance(var.value, Expression):
                    dependent_expressions.add(str(var.value))
                    _get_dependent_expressions(var.value, dependent_expressions)
            except ValueError:
                # An undefined variable is found. Validation will handle this.
                pass

    def _collect_expressions_recursive(data, used_expressions: set):
        """Recursively collect expressions from nested data structures."""
        if isinstance(data, dict):
            # Check if this dict is a UserVariable
            if _is_user_variable(data):
                variable_name = data.get("name", {})
                if "." in variable_name:
                    return
                try:
                    value = default_context.get(variable_name)
                    if isinstance(value, Expression):
                        used_expressions.add(str(value))
                except ValueError:
                    # An undefined variable is found. Validation will handle this.
                    pass

            # Check if this dict is an Expression
            elif _is_expression(data):
                usage = _is_expression(data)
                if usage == ExpressionUsage.VALUE_OR_EXPRESSION:
                    used_expressions.add(data.get("expression"))
                elif usage == ExpressionUsage.EXPRESSION_AS_IS:
                    used_expressions.add(data.get("expression"))

            # Recursively process all values in the dict
            for value in data.values():
                _collect_expressions_recursive(value, used_expressions)

        elif isinstance(data, list):
            # Recursively process all items in the list
            for item in data:
                _collect_expressions_recursive(item, used_expressions)

    if (
        "private_attribute_asset_cache" not in param_as_dict
        or "variable_context" not in param_as_dict["private_attribute_asset_cache"]
    ):
        return [], []

    used_expressions: set[str] = set()
    param_as_dict_without_project_variables = copy.deepcopy(param_as_dict)
    param_as_dict_without_project_variables["private_attribute_asset_cache"][
        "variable_context"
    ] = []

    _collect_expressions_recursive(param_as_dict_without_project_variables, used_expressions)

    dependent_expressions = set()

    for expr in used_expressions:
        _get_dependent_expressions(Expression(expression=expr), dependent_expressions)

    return list(used_expressions.union(dependent_expressions))
