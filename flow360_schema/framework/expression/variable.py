"""This module allows users to write serializable, evaluable symbolic code for use in simulation params"""

from __future__ import annotations

import ast
import contextlib
import logging
import re
import textwrap
from collections.abc import Callable
from numbers import Number
from typing import (
    Any,
    Literal,
)

import numpy as np
import pydantic as pd
import unyt as u
from pydantic_core import InitErrorDetails
from typing_extensions import Self
from unyt import Unit, dimensions, unyt_array, unyt_quantity

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.expression.engine.eval_context import EvaluationContext
from flow360_schema.framework.expression.engine.generator import expr_to_code
from flow360_schema.framework.expression.engine.parser import expr_to_model
from flow360_schema.framework.expression.engine.types import Evaluable, TargetSyntax
from flow360_schema.framework.expression.registry import (
    LEGACY_OUTPUT_FIELD_NAMES,
    SOLVER_INTERNAL_VARIABLES,
    default_context,
)
from flow360_schema.framework.expression.utils import (
    handle_syntax_error,
    is_number_string,
    split_keep_delimiters,
)
from flow360_schema.framework.validation.validators import contextual_model_validator as _contextual_model_validator
from flow360_schema.models.variable_context import VariableContextInfo as _VariableContextInfo

VariableContextInfo = _VariableContextInfo

logger = logging.getLogger(__name__)


class RedeclaringVariableError(ValueError):
    """Raised when a user variable is redeclared with a different value."""

    def __init__(self, variable_name: str, new_value: Any, previous_value: Any) -> None:
        self.variable_name = variable_name
        self.new_value = new_value
        self.previous_value = previous_value
        super().__init__(
            f"Redeclaring user variable '{variable_name}' with new value: {new_value}. "
            f"Previous value: {previous_value}"
        )


_solver_variables: dict[str, str] = {}


def get_user_variable(name: str) -> UserVariable:
    """Get the user variable from the global context."""
    return UserVariable(name=name, value=default_context.get(name))  # type: ignore[call-arg]


def remove_user_variable(name: str) -> None:
    """Remove the variable from the global context."""
    return default_context.remove(name)


def show_user_variables() -> None:
    """Show the user variables from the global context with name and value in two columns, wrapping long values."""

    user_variables = {name: default_context.get(name) for name in sorted(default_context.user_variable_names)}

    if not user_variables.keys():
        logger.info("No user variables are currently defined.")
        return

    header_index = "Idx"
    header_name = "Name"
    header_value = "Value"

    max_name_width = max(max(len(name) for name in user_variables), len(header_name))

    terminal_width = 100

    value_col_width = max(terminal_width - (len(header_index) + 1 + max_name_width), 20)

    formatted_header = f"{header_index:>{len(header_index)}}. " f"{header_name:<{max_name_width}} " f"{header_value}"
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

        output_lines.append(f"{idx+1:>{len(header_index)}}. {name:<{max_name_width}} {first_value_line}")

        indent_for_wrapped_lines = " " * (len(header_index) + max_name_width + 2)

        for subsequent_line in wrapped_value_lines[1:]:
            output_lines.append(f"{indent_for_wrapped_lines}{subsequent_line}")

    output_text = "\n".join(output_lines)

    logger.info("The current defined user variables are:\n%s", output_text)


def __soft_fail_add__(self: object, other: object) -> object:
    import numpy as np

    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__add__(self, other)  # type: ignore[operator]
    return NotImplemented


def __soft_fail_sub__(self: object, other: object) -> object:
    import numpy as np

    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__sub__(self, other)  # type: ignore[operator]
    return NotImplemented


def __soft_fail_mul__(self: object, other: object) -> object:
    import numpy as np

    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__mul__(self, other)  # type: ignore[operator]
    return NotImplemented


def __soft_fail_truediv__(self: object, other: object) -> object:
    import numpy as np

    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__truediv__(self, other)  # type: ignore[operator]
    return NotImplemented


def __soft_fail_pow__(self: object, other: object) -> object:
    import numpy as np

    if not isinstance(other, Expression) and not isinstance(other, Variable):
        return np.ndarray.__pow__(self, other)  # type: ignore[operator]
    return NotImplemented


def _patch_unyt_array() -> None:
    """Apply monkey-patches to unyt_array arithmetic to support Expression/Variable interop."""
    unyt_array.__add__ = __soft_fail_add__
    unyt_array.__sub__ = __soft_fail_sub__
    unyt_array.__mul__ = __soft_fail_mul__
    unyt_array.__truediv__ = __soft_fail_truediv__
    unyt_array.__pow__ = __soft_fail_pow__


_patch_unyt_array()


def _convert_numeric(value: Number | Unit | unyt_array | list[Any]) -> str | None:
    arg = None
    unit_delimiters = ["+", "-", "*", "/", "(", ")"]
    if isinstance(value, Number):
        arg = str(value)
    elif isinstance(value, u.Unit):
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


def _convert_argument(value: Any) -> tuple[str, bool]:
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


def _is_array(item: Any) -> bool:
    if isinstance(item, unyt_array) and item.shape != ():
        return True
    if isinstance(item, list):
        return True
    return False


def check_vector_binary_arithmetic(func: Callable[..., Expression]) -> Callable[..., Expression]:
    """Decorator to check if vector arithmetic is being attempted and raise an error if so."""

    def wrapper(self: Variable, other: Any) -> Expression:
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
    def value(self) -> Any:
        """
        Get the value of the variable from the global context.
        """
        return default_context.get(self.name)

    @value.setter
    def value(self, value: Any) -> None:
        """
        Set the value of the variable in the global context.
        In parallel to `deserialize` this supports syntax like `my_user_var.value = 10.0`.
        """
        from flow360_schema.framework.expression.value_or_expression import (
            AnyNumericType,
            ValueOrExpression,
        )

        value = pd.TypeAdapter(
            ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore[index]
        ).validate_python(value)
        # Not checking overwrite here since it is user controlled explicit assignment operation
        default_context.set_value(self.name, value)

    @pd.model_validator(mode="before")
    @classmethod
    def preprocess_variable_declaration(cls, values: Any) -> dict[str, Any]:
        """
        Supporting syntax like `a = fl.Variable(name="a", value=1, description="some description")`.
        """
        # Pass through existing Variable instances (e.g. schema UserVariable used in client context)
        if isinstance(values, Variable):
            return {"name": values.name}
        if values is None or "name" not in values:
            raise ValueError("`name` is required for variable declaration.")

        if "value" in values:
            raw_value = values.pop("value")

            # Solver variables (names with ".") store raw values without Expression conversion.
            # NaN values in solver variable definitions can't be expressed as evaluable Expressions.
            if "." in values["name"]:
                default_context.set_value(values["name"], raw_value)
            else:
                from flow360_schema.framework.expression.value_or_expression import (
                    AnyNumericType,
                    ValueOrExpression,
                )

                new_value = pd.TypeAdapter(
                    ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore[index]
                ).validate_python(raw_value)

                # Check redeclaration:
                if values["name"] in default_context.user_variable_names:
                    registered_expression = _convert_variable_value_to_expression(default_context.get(values["name"]))
                    registered_expression_stripped = registered_expression.expression.replace(" ", "")

                    if isinstance(new_value, Expression):
                        new_value_stripped = new_value.expression.replace(" ", "")
                    else:
                        new_value_stripped = _convert_variable_value_to_expression(new_value).expression.replace(
                            " ", ""
                        )

                    if new_value_stripped != registered_expression_stripped:
                        raise RedeclaringVariableError(
                            variable_name=values["name"],
                            new_value=new_value,
                            previous_value=default_context.get(values["name"]),
                        )
                else:
                    # No conflict, call the setter
                    default_context.set_value(
                        values["name"],
                        new_value,
                    )

        if values.get("description") is not None:
            if not isinstance(values["description"], str):
                raise ValueError(f"Description must be a string but got {type(values['description'])}.")
            default_context.set_metadata(values["name"], "description", values["description"])
        values.pop("description", None)

        if values.get("metadata") is not None:
            default_context.set_metadata(values["name"], "metadata", values["metadata"])
        values.pop("metadata", None)
        return values  # type: ignore[no-any-return]

    @check_vector_binary_arithmetic
    def __add__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} + {str_arg}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __sub__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} - {str_arg}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __mul__(self, other: Any) -> Expression:
        if isinstance(other, Number) and other == 0:  # type: ignore[comparison-overlap]
            return Expression(expression="0")  # type: ignore[call-arg]

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} * {str_arg}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __truediv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} / ({str_arg})")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __floordiv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} // ({str_arg})")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __mod__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.name} % {str_arg}")  # type: ignore[call-arg]

    def __pow__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        if _is_array(self.value):
            components = [f"{self.name}[{i}] ** {str_arg}" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")  # type: ignore[call-arg]
        return Expression(expression=f"{self.name} ** {str_arg}")  # type: ignore[call-arg]

    def __neg__(self) -> Expression:
        if _is_array(self.value):
            components = [f"-{self.name}[{i}]" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")  # type: ignore[call-arg]
        return Expression(expression=f"-{self.name}")  # type: ignore[call-arg]

    def __pos__(self) -> Expression:
        if _is_array(self.value):
            components = [f"+{self.name}[{i}]" for i in range(len(self.value))]
            return Expression(expression=f"[{','.join(components)}]")  # type: ignore[call-arg]
        return Expression(expression=f"+{self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __radd__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rsub__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rmul__(self, other: Any) -> Expression:
        if isinstance(other, Number) and other == 0:  # type: ignore[comparison-overlap]
            return Expression(expression="0")  # type: ignore[call-arg]

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rtruediv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rfloordiv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rmod__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % {self.name}")  # type: ignore[call-arg]

    @check_vector_binary_arithmetic
    def __rpow__(self, other: Any) -> Expression:
        (arg, _) = _convert_argument(other)
        str_arg = f"({arg})"  # Always parenthesize to ensure base is evaluated first
        return Expression(expression=f"{str_arg} ** {self.name}")  # type: ignore[call-arg]

    def __getitem__(self, item: Any) -> Expression:
        (arg, _) = _convert_argument(item)
        return Expression(expression=f"{self.name}[{arg}]")  # type: ignore[call-arg]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Variable({self.name} = {self.value})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        # NaN-compatible equal operator for unit test support
        if not isinstance(other, Variable):
            return False
        return self.model_dump_json() == other.model_dump_json()

    def __len__(self) -> int:
        """The number of elements in self.value. 0 for scalar and anything else for vector."""
        if isinstance(self.value, Expression):
            return len(self.value)
        if isinstance(self.value, unyt_array):
            # Can be either unyt_array or unyt_quantity
            if self.value.shape == ():
                return 0
            # No 2D arrays are supported
            return int(self.value.shape[0])
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
    def check_valid_user_variable_name(cls, v: str) -> str:
        """Validate a variable identifier (ASCII only)."""
        # Partial list of keywords; extend as needed
        RESERVED_SYNTAX_KEYWORDS = {
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
            raise ValueError("Identifier can only contain letters, digits (0-9), or underscore (_).")

        # 4) Not a C++ keyword
        if v in RESERVED_SYNTAX_KEYWORDS:
            raise ValueError(f"'{v}' is a reserved keyword.")

        # 5) Legacy output field names — checked first so the error message is specific
        # TODO(deprecation migration): Add deprecation_reminder decorator once Flow360Version is available in schema.
        if v in LEGACY_OUTPUT_FIELD_NAMES:
            raise ValueError(f"'{v}' is a reserved (legacy) output field name. It cannot be used in expressions.")

        # 6) Solver-side variable names
        solver_side_names = {item.split(".")[-1] for item in default_context.registered_names if "." in item}
        solver_side_names = solver_side_names.union(SOLVER_INTERNAL_VARIABLES)
        if v in solver_side_names:
            raise ValueError(f"'{v}' is a reserved solver side variable name.")

        return v

    def __hash__(self) -> int:
        """
        Support for set and deduplicate.
        """
        return hash(self.model_dump_json())

    def in_units(
        self,
        new_unit: str | Literal["SI_unit_system", "CGS_unit_system", "Imperial_unit_system"] | Unit = None,
    ) -> Self:
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
    solver_name: str | None = pd.Field(None)
    variable_type: Literal["Volume", "Surface", "Scalar"] = pd.Field()

    @pd.model_validator(mode="after")
    def update_context(self) -> Self:
        """Auto updating context when new variable is declared"""
        default_context.set_value(self.name, self.value, Variable)  # type: ignore[arg-type]
        _solver_variables.update({self.name: self.variable_type})
        if self.solver_name:
            default_context.set_alias(self.name, self.solver_name)
        return self

    def in_units(
        self,
        new_name: str,
        new_unit: str | Literal["SI_unit_system", "CGS_unit_system", "Imperial_unit_system"] | Unit = None,
    ) -> UserVariable:
        """
        Return a UserVariable that will generate results in the new_unit.
        If new_unit is not specified then the unit will be determined by the unit system.
        """
        if isinstance(new_unit, Unit):
            new_unit = str(new_unit)
        new_variable = UserVariable(
            name=new_name,
            value=Expression(expression=self.name),  # type: ignore[call-arg]
        )
        new_variable.value.output_units = new_unit
        return new_variable


def get_input_value_length(
    value: Number | list[float] | Expression | Variable,
) -> int:
    """Get the length of the input value."""
    if isinstance(value, Expression):
        evaluated: Any = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
    else:
        evaluated = value
    assert isinstance(
        evaluated, (unyt_array, unyt_quantity, list, Number, np.ndarray)
    ), f"Unexpected evaluated result type: {type(evaluated)}"
    if isinstance(evaluated, list):
        return len(evaluated)
    if isinstance(evaluated, np.ndarray):
        return 0 if evaluated.shape == () else evaluated.shape[0]
    return 0 if isinstance(evaluated, (unyt_quantity, Number)) else evaluated.shape[0]


_feature_requirement_map: dict[str, tuple[Any, str]] = {
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
        lambda x: x.time_stepping.value != "Steady",
        "Unsteady time stepping is not used.",
    ),
    "control.timeStepSize": (
        lambda x: x.time_stepping.value != "Steady",
        "Unsteady time stepping is not used.",
    ),
    "control.theta": (
        lambda x: x.feature_usage.rotation_zone_count > 0,
        "Rotation zone is not used.",
    ),
    "control.omega": (
        lambda x: x.feature_usage.rotation_zone_count > 0,
        "Rotation zone is not used.",
    ),
    "control.omegaDot": (
        lambda x: x.feature_usage.rotation_zone_count > 0,
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
    output_units: str | None = pd.Field(
        None,
        description="String representation of what the requested units the evaluated expression should be "
        "when `self` is used as an output field. By default the output units will be inferred from the unit "
        "system associated with SimulationParams",
    )

    model_config = pd.ConfigDict(validate_assignment=True)

    @pd.model_validator(mode="before")
    @classmethod
    def _validate_expression(cls, value: Any) -> dict[str, Any]:
        output_units = None
        if isinstance(value, str):
            expression = value
        elif isinstance(value, dict) and "expression" in value:
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
        elif isinstance(value, (Number, u.unyt_array, u.unyt_quantity)):
            converted = _convert_numeric(value)
            assert converted is not None, f"Failed to convert numeric value: {value}"
            expression = converted
        else:
            details = InitErrorDetails(type="value_error", ctx={"error": f"Invalid type {type(value)}"}, input="")
            raise pd.ValidationError.from_exception_data("Expression type error", [details])

        try:
            # To ensure the expression is valid (also checks for
            expr_to_model(expression, default_context)
            # To reduce unnecessary parentheses
            expression = ast.unparse(ast.parse(expression))
        except SyntaxError as s_err:
            handle_syntax_error(s_err, expression)
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err}, input="")
            raise pd.ValidationError.from_exception_data("Expression value error", [details]) from v_err

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
            raise ValueError("^ operator is not allowed in expressions. For power operator, please use ** instead.")
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
            raise ValueError("Relative temperature scale usage is not allowed. Please use u.R or u.K instead.")
        return value

    @pd.model_validator(mode="after")
    def check_output_units_matches_dimensions(self) -> Self:
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

    @_contextual_model_validator(mode="after")
    def ensure_dependent_feature_enabled(self, param_info: Any) -> Expression:
        """Ensure that all dependent features are enabled for the solver variables used."""
        if self.expression not in param_info.referenced_expressions:
            return self
        # Setting recursive to False to avoid recursive error message.
        # All user variables will be checked anyways.
        for solver_variable_name in self.solver_variable_names(recursive=False):
            if solver_variable_name in _feature_requirement_map and not _feature_requirement_map[solver_variable_name][
                0
            ](param_info):
                raise ValueError(
                    f"`{solver_variable_name}` cannot be used "
                    f"because {_feature_requirement_map[solver_variable_name][1]}"
                )
        return self

    def evaluate(
        self,
        context: EvaluationContext | None = None,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> float | list[Any] | unyt_array | Expression:
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

    def user_variables(self) -> list[UserVariable]:
        """Get list of user variables used in expression."""
        expr = expr_to_model(self.expression, default_context)
        all_names = expr.used_names()
        filtered_names = [name for name in all_names if name in default_context.user_variable_names]

        return [UserVariable(name=name, value=default_context.get(name)) for name in filtered_names]  # type: ignore[call-arg]

    def user_variable_names(self) -> list[str]:
        """Get list of user variable names used in expression."""
        expr = expr_to_model(self.expression, default_context)
        all_names = expr.used_names()
        filtered_names = [name for name in all_names if name in default_context.user_variable_names]

        return filtered_names

    def solver_variable_names(
        self,
        recursive: bool,
        variable_type: Literal["Volume", "Surface", "Scalar", "All"] = "All",
    ) -> list[str]:
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
            solver_names: set[str] = set()

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
            all_solver_names = {name for name in all_solver_names if _solver_variables[name] == variable_type}

        return list(all_solver_names)

    def to_solver_code(self, unit_system: Any) -> str:
        """Convert to solver readable code."""

        def translate_symbol(name: str) -> str:
            alias = default_context.get_alias(name)

            if alias:
                return alias

            match = re.fullmatch("u\\.(.+)", name)

            if match:
                unit_name = match.group(1)
                unit = Unit(unit_name)
                if unit == u.dimensionless:
                    return "1.0"
                conversion_factor = (1.0 * unit).in_base(unit_system=unit_system).v
                return str(conversion_factor)

            # solver-time resolvable functions:
            func_match = re.fullmatch(r"math\.(.+)", name)
            if func_match:
                func_name = func_match.group(1)
                return func_name

            return name

        partial_result = self.evaluate(default_context, raise_on_non_evaluable=False, force_evaluate=False)

        if isinstance(partial_result, Expression):
            expr = expr_to_model(partial_result.expression, default_context)
        else:
            numeric_str = _convert_numeric(partial_result)
            assert numeric_str is not None, f"Failed to convert numeric result: {partial_result}"
            expr = expr_to_model(numeric_str, default_context)
        return expr_to_code(expr, TargetSyntax.CPP, translate_symbol)

    def __hash__(self) -> int:
        return hash(self.expression)

    def __add__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.expression} + {str_arg}")  # type: ignore[call-arg]

    def __sub__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{self.expression} - {str_arg}")  # type: ignore[call-arg]

    def __mul__(self, other: Any) -> Expression:
        if isinstance(other, Number) and other == 0:  # type: ignore[comparison-overlap]
            return Expression(expression="0")  # type: ignore[call-arg]

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) * {str_arg}")  # type: ignore[call-arg]

    def __truediv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) / ({str_arg})")  # type: ignore[call-arg]

    def __floordiv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) // ({str_arg})")  # type: ignore[call-arg]

    def __mod__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) % {str_arg}")  # type: ignore[call-arg]

    def __pow__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"({self.expression}) ** {str_arg}")  # type: ignore[call-arg]

    def __neg__(self) -> Expression:
        return Expression(expression=f"-({self.expression})")  # type: ignore[call-arg]

    def __pos__(self) -> Expression:
        return Expression(expression=f"+({self.expression})")  # type: ignore[call-arg]

    def __abs__(self) -> Expression:
        return Expression(expression=f"abs({self.expression})")  # type: ignore[call-arg]

    def __radd__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} + {self.expression}")  # type: ignore[call-arg]

    def __rsub__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} - {self.expression}")  # type: ignore[call-arg]

    def __rmul__(self, other: Any) -> Expression:
        if isinstance(other, Number) and other == 0:  # type: ignore[comparison-overlap]
            return Expression(expression="0")  # type: ignore[call-arg]

        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} * ({self.expression})")  # type: ignore[call-arg]

    def __rtruediv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} / ({self.expression})")  # type: ignore[call-arg]

    def __rfloordiv__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} // ({self.expression})")  # type: ignore[call-arg]

    def __rmod__(self, other: Any) -> Expression:
        (arg, parenthesize) = _convert_argument(other)
        str_arg = arg if not parenthesize else f"({arg})"
        return Expression(expression=f"{str_arg} % ({self.expression})")  # type: ignore[call-arg]

    def __rpow__(self, other: Any) -> Expression:
        (arg, _) = _convert_argument(other)
        str_arg = f"({arg})"  # Always parenthesize to ensure base is evaluated first
        return Expression(expression=f"{str_arg} ** ({self.expression})")  # type: ignore[call-arg]

    def __getitem__(self, index: Any) -> Expression:
        (arg, _) = _convert_argument(index)
        tree = ast.parse(self.expression, mode="eval")
        int_arg = None
        with contextlib.suppress(ValueError):
            int_arg = int(arg)
        if isinstance(tree.body, ast.List) and int_arg is not None:
            # Expression string with list syntax, like "[aa,bb,cc]"
            # and since the index is static we can reduce it
            result = [ast.unparse(elt) for elt in tree.body.elts]
            return Expression.model_validate(result[int_arg])
        return Expression(expression=f"({self.expression})[{arg}]")  # type: ignore[call-arg]

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return f"Expression({self.expression})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expression):
            return self.expression == other.expression
        return super().__eq__(other)

    @property
    def dimensions(self) -> Any:
        """The physical dimensions of the expression."""
        value = self.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
        if isinstance(value, (unyt_array, unyt_quantity)):
            return value.units.dimensions
        if isinstance(value, list):
            _check_list_items_are_same_dimensions(value)
            # Check unyt first — unyt_quantity is a subclass of np.ndarray
            if isinstance(value[0], (unyt_array, unyt_quantity)):
                return value[0].units.dimensions
            if isinstance(value[0], (Number, np.ndarray)):
                return u.Unit("dimensionless").dimensions
        if isinstance(value, (Number, np.ndarray)):
            # Plain numbers or numpy arrays without units are dimensionless
            return u.Unit("dimensionless").dimensions
        raise ValueError(f"Cannot determine dimensions for expression with type {type(value).__name__}: {value}")

    @property
    def length(self) -> int:
        """The number of elements in the expression. 0 for scalar and anything else for vector."""
        return get_input_value_length(self)

    def __len__(self) -> int:
        return self.length

    def get_output_units(self, unit_system_name: str | None = None) -> Unit:
        """
        Get the output units of the expression.

        - If self.output_units is None, derive the default output unit based on the
        value's dimensions and current unit system.

        - If self.output_units is valid u.Unit string, deserialize it and return it.

        - If self.output_units is valid unit system name, derive the default output
        unit based on the value's dimensions and the **given** unit system.

        - If expression is a number constant, return None.

        - Else raise ValueError.

        Args:
            unit_system_name: Name of the unit system (e.g. "SI", "Imperial", "CGS").
                Required when self.output_units is not a valid u.Unit string.
        """

        def _get_unit_from_unit_system(expression: Expression, name: str) -> Unit:
            """Derive the default output unit based on the value's dimensions and current unit system"""
            numerical_value = expression.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            if isinstance(numerical_value, list):
                numerical_value = numerical_value[0]
            if not isinstance(numerical_value, (u.unyt_array, u.unyt_quantity)):
                # Pure dimensionless constant
                return u.Unit("dimensionless")
            if name in ("SI", "SI_unit_system"):
                return numerical_value.in_base("mks").units
            if name in ("Imperial", "Imperial_unit_system"):
                return numerical_value.in_base("imperial").units
            if name in ("CGS", "CGS_unit_system"):
                return numerical_value.in_base("cgs").units
            raise ValueError(f"[Internal] Invalid unit system: {name}")

        try:
            return u.Unit(self.output_units)
        except u.exceptions.UnitParseError as e:
            if not self.output_units:
                if unit_system_name is None:
                    raise ValueError("[Internal] unit_system_name required when output_units is not set") from e
            else:
                unit_system_name = self.output_units
            # The unit system for inferring the units for input has different temperature unit
            u.unit_systems.imperial_unit_system["temperature"] = u.Unit("R").expr
            result = _get_unit_from_unit_system(self, unit_system_name)
            u.unit_systems.imperial_unit_system["temperature"] = u.Unit("degF").expr
            return result


def _check_list_items_are_same_dimensions(value: list[Any]) -> None:
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
    if any(isinstance(item, Number) for item in value) and any(isinstance(item, unyt_quantity) for item in value):
        raise ValueError("List must contain only all unyt_quantities or all numbers.")
    return


def get_input_value_dimensions(
    value: float | list[float] | Expression | Variable,
) -> Any:
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


def _convert_variable_value_to_expression(value: Any) -> Any:
    """Normalize variable-context values to Expression for runtime consistency."""
    if isinstance(value, Expression):
        return value
    return Expression.model_validate(_convert_numeric(value))


def _patch_variable_context_info_value_field() -> None:
    """Replace VariableContextInfo.value annotation with the real ValueOrExpression type.

    Must be called after all expression modules are fully loaded (from __init__.py)
    to avoid circular imports.
    """
    if VariableContextInfo.model_fields["value"].annotation is not Any:
        return  # already patched
    from flow360_schema.framework.expression.value_or_expression import (
        AnyNumericType,
        ValueOrExpression,
    )

    VariableContextInfo.model_fields["value"].annotation = (
        ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore[index]
    )
    VariableContextInfo.model_rebuild(force=True)


def save_user_variables(
    variable_context: list[VariableContextInfo] | None,
    post_processing_names: set[str],
    output_units: dict[str, str],
) -> list[VariableContextInfo]:
    """Save user variables to variable context list.

    Args:
        variable_context: Existing variable context (may be None).
        post_processing_names: Set of variable names used in post-processing outputs.
        output_units: Dict mapping variable name -> output unit string.

    Returns:
        Updated list of VariableContextInfo.
    """
    if variable_context is None:
        variable_context = []

    user_variable_names = default_context.user_variable_names
    for name, value in list(default_context._values.items()):
        if name not in user_variable_names:
            continue

        output_unit_str = output_units.get(name)
        if output_unit_str is not None:
            if isinstance(value, Expression):
                value = value.model_copy()
                value.output_units = output_unit_str
            else:
                value = _convert_variable_value_to_expression(value)
                value.output_units = output_unit_str

        existing_index = None
        for i, existing_var in enumerate(variable_context):
            if existing_var.name == name:
                existing_index = i
                break

        new_variable = VariableContextInfo(
            name=name,
            value=value,
            description=default_context.get_metadata(name, "description"),
            post_processing=name in post_processing_names,
            metadata=default_context.get_metadata(name, "metadata"),
        )

        if existing_index is not None:
            variable_context[existing_index] = new_variable
        else:
            variable_context.append(new_variable)

    return variable_context


def batch_get_user_variable_units(
    variable_names: list[str],
    unit_system_name: str,
) -> dict[str, Any]:
    """Return output units for a list of user variable names.

    For each name, the value is pulled from default_context and converted to a unit:
    - Expression: Expression.get_output_units(unit_system_name)
    - unyt_array/unyt_quantity: their units
    - Number: "dimensionless"

    Args:
        variable_names: List of user variable names to look up.
        unit_system_name: Unit system name (e.g. "SI") for Expression.get_output_units.

    Returns:
        Dict mapping variable name to a unyt.Unit or the string "dimensionless".

    Raises:
        ValueError: If a name resolves to an unsupported type.
    """
    result = {}
    for name in variable_names:
        value = default_context.get(name)
        if isinstance(value, Expression):
            result[name] = value.get_output_units(unit_system_name=unit_system_name)
        elif isinstance(value, u.unyt_array):
            result[name] = value.units
        elif isinstance(value, Number):
            result[name] = "dimensionless"
        else:
            raise ValueError(f"Unknown variable type: {value}")
    return result


def compute_surface_integral_unit(
    variable: UserVariable,
    unit_system_name: str,
    unit_system: Any,
) -> str:
    """Compute the unit of the surface integral of a UserVariable over a surface.

    Args:
        variable: The UserVariable to compute units for.
        unit_system_name: Unit system name (e.g. "SI") for get_output_units.
        unit_system: unyt.UnitSystem instance for area unit lookup.
    """
    base_unit = None
    if isinstance(variable.value, Expression):
        base_unit = variable.value.get_output_units(unit_system_name=unit_system_name)
    else:
        val = variable.value
        if hasattr(val, "get_output_units"):
            base_unit = val.get_output_units(unit_system_name=unit_system_name)
        elif isinstance(val, (u.unyt_array, u.unyt_quantity)):
            base_unit = val.units
        else:
            base_unit = u.Unit("dimensionless")

    if base_unit is None:
        base_unit = u.Unit("dimensionless")

    area_unit = unit_system["area"].units
    result_unit = base_unit * area_unit
    return str(result_unit)


def restore_variable_space(
    variable_context: list[dict[str, Any]],
    clear_first: bool = False,
) -> None:
    """Restore variable space from serialized variable context.

    Topologically sorts variables by dependency, then creates UserVariable instances
    (which auto-register into default_context).

    Args:
        variable_context: List of serialized variable dicts (each has name, value, etc.)
        clear_first: If True, clear existing user variables before restoring.

    Raises:
        pydantic.ValidationError: If a variable is invalid or redeclares an existing one.
    """
    from flow360_schema.framework.expression.dependency_graph import DependencyGraph
    from flow360_schema.framework.expression.registry import clear_context

    if clear_first:
        clear_context()

    dependency_graph = DependencyGraph()
    variable_list = []
    for var in variable_context:
        val = var["value"]
        if val.get("type_name") == "expression" or ("expression" in val and "value" not in val):
            variable_list.append({"name": var["name"], "value": val["expression"]})
        else:
            variable_list.append({"name": var["name"], "value": str(val["value"])})
    dependency_graph.load_from_list(variable_list)
    sorted_variables = dependency_graph.topology_sort()

    name_to_index = {var["name"]: idx for idx, var in enumerate(variable_context)}

    for variable_name in sorted_variables:
        variable_dict = next(
            (var for var in variable_context if var["name"] == variable_name),
            None,
        )
        if variable_dict is None:
            continue

        value_or_expression = dict(variable_dict["value"].items())
        # Pre-migration UserVariable parsed value manually without a discriminator.
        # Schema's ValueOrExpression uses type_name as discriminator, so we infer
        # it from the dict shape when absent (e.g. legacy serialized data).
        if "type_name" not in value_or_expression:
            if "expression" in value_or_expression:
                value_or_expression["type_name"] = "expression"
            else:
                value_or_expression["type_name"] = "number"

        try:
            UserVariable(
                name=variable_dict["name"],
                value=value_or_expression,
                description=variable_dict.get("description", None),
                metadata=variable_dict.get("metadata", None),
            )  # type: ignore[call-arg]  # Pydantic dynamic fields
        except pd.ValidationError as e:
            # Unwrap RedeclaringVariableError from Pydantic wrapping.
            # Pydantic stores the original exception in ctx['error'].
            error_detail = e.errors()[0]
            original_error = error_detail.get("ctx", {}).get("error")
            if isinstance(original_error, RedeclaringVariableError):
                raise original_error from e
            raise pd.ValidationError.from_exception_data(
                "Invalid user variable/expression",
                line_errors=[
                    {
                        "type": error_detail["type"],
                        "loc": (
                            "variable_context",
                            name_to_index.get(variable_name, 0),
                        ),
                        "input": "",
                        "ctx": error_detail.get("ctx", {}),
                    },
                ],
            ) from e


def get_referenced_expressions_and_user_variables(param_as_dict: dict[str, Any]) -> list[str]:
    """Get all expressions referenced in the params dict (excluding asset cache).

    Two sources:
    1. Field is Expression.
    2. Field is UserVariable and value is an Expression.
    Expression and UserVariable are both identified by their schema.
    """

    def _is_user_variable(field: dict[str, Any]) -> bool:
        return "type_name" in field and field["type_name"] == "UserVariable"

    def _is_expression(field: dict[str, Any]) -> bool:
        if "type_name" in field and field["type_name"] == "expression":
            return True
        if sorted(field.keys()) == ["expression", "output_units"] or sorted(field.keys()) == ["expression"]:
            return True
        return False

    def _get_dependent_expressions(
        expression: Expression,
        dependent_expressions: set[str],
    ) -> None:
        """Get all the expressions that are dependent on the given expression."""
        for var in expression.user_variables():
            try:
                if "." not in var.name and isinstance(var.value, Expression):
                    dependent_expressions.add(str(var.value))
                    _get_dependent_expressions(var.value, dependent_expressions)
            except ValueError:
                # An undefined variable is found. Validation will handle this.
                pass

    def _collect_expressions_recursive(
        data: Any,
        used_expressions: set[str],
        current_path: tuple[str, ...] = (),
        exclude_paths: set[tuple[str, ...]] | None = None,
    ) -> None:
        """Recursively collect expressions from nested data structures."""
        if exclude_paths is None:
            exclude_paths = {("private_attribute_asset_cache", "variable_context")}

        if data is None or isinstance(data, (int, float, str, bool)):
            return

        if current_path in exclude_paths:
            return

        if isinstance(data, dict):
            if _is_user_variable(data):
                variable_name = data.get("name", {})
                if "." in variable_name:
                    return
                try:
                    value = default_context.get(variable_name)
                    if isinstance(value, Expression):
                        used_expressions.add(str(value))
                except ValueError:
                    pass

            elif _is_expression(data):
                expr_str = data.get("expression")
                if expr_str is not None:
                    used_expressions.add(expr_str)

            for key, value in data.items():
                _collect_expressions_recursive(
                    value,
                    used_expressions,
                    current_path + (key,),
                    exclude_paths,
                )

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                _collect_expressions_recursive(
                    item,
                    used_expressions,
                    current_path + (str(idx),),
                    exclude_paths,
                )

    if (
        "private_attribute_asset_cache" not in param_as_dict
        or "variable_context" not in param_as_dict["private_attribute_asset_cache"]
    ):
        return []

    used_expressions: set[str] = set()
    _collect_expressions_recursive(
        param_as_dict,
        used_expressions,
        current_path=(),
        exclude_paths={("private_attribute_asset_cache", "variable_context")},
    )

    dependent_expressions: set[str] = set()
    for expr in used_expressions:
        _get_dependent_expressions(Expression(expression=expr), dependent_expressions)  # type: ignore[call-arg]

    return list(used_expressions.union(dependent_expressions))


def solver_variable_to_user_variable(item: Variable) -> Variable:
    """Convert a SolverVariable to a UserVariable using the current unit system."""
    from flow360_schema.framework.validation.context import unit_system_manager

    if isinstance(item, SolverVariable):
        if unit_system_manager.current is None:
            raise ValueError(f"Solver variable {item.name} cannot be used without a unit system.")
        unit_system_name = unit_system_manager.current.name
        name = item.name.split(".")[-1] if "." in item.name else item.name
        user_var = UserVariable(name=f"{name}_{unit_system_name}", type_name="UserVariable")
        user_var.value = item
        return user_var
    return item
