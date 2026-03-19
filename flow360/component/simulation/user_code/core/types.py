# pylint: disable=too-many-lines
"""This module allows users to write serializable, evaluable symbolic code for use in simulation params.

Core types (Variable, UserVariable, SolverVariable, Expression) live in flow360-schema.
This file re-imports them, adds client-only validators via subclassing, and keeps
items that have not yet migrated (ValueOrExpression, VariableContextInfo, etc.).
"""

from __future__ import annotations

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
from flow360_schema import StrictUnitContext
from flow360_schema.framework.expression.registry import default_context
from flow360_schema.framework.expression.utils import is_runtime_expression

# pylint: disable=unused-import
from flow360_schema.framework.expression.variable import Expression as _SchemaExpression
from flow360_schema.framework.expression.variable import (  # noqa: F401 (re-export); noqa: F401 (re-exports)
    SolverVariable,
)
from flow360_schema.framework.expression.variable import (
    UserVariable as _SchemaUserVariable,
)
from flow360_schema.framework.expression.variable import Variable as _SchemaVariable
from flow360_schema.framework.expression.variable import (  # noqa: F401 (re-export); noqa: F401 (re-exports)
    _check_list_items_are_same_dimensions,
    _convert_argument,
    _convert_numeric,
    _is_array,
    _solver_variables,
    check_vector_binary_arithmetic,
    get_input_value_dimensions,
    get_input_value_length,
    get_user_variable,
    remove_user_variable,
    show_user_variables,
)

# pylint: enable=unused-import
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from pydantic_core import core_schema
from typing_extensions import Self
from unyt import unyt_array, unyt_quantity

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater_utils import deprecation_reminder
from flow360.component.simulation.unit_system import unit_system_manager
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_model_validator,
)

# ---------------------------------------------------------------------------
# Client subclass: Variable — restores ValueOrExpression in setter/validator
# ---------------------------------------------------------------------------

# MIGRATION-TODO(validation framework migration): Remove this subclass once
#   ValueOrExpression is migrated to schema (phase 3).


class Variable(_SchemaVariable):
    """Client-side Variable with ValueOrExpression support in value setter and declaration."""

    @property
    def value(self):
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
        default_context.set_value(self.name, new_value)

    @pd.model_validator(mode="before")
    @classmethod
    def preprocess_variable_declaration(cls, values):
        """
        Supporting syntax like `a = fl.Variable(name="a", value=1, description="some description")`.
        """
        # Pass through existing Variable instances (e.g. schema UserVariable used in client context)
        # TOAI: I am confused, why do we need this change?
        if isinstance(values, _SchemaVariable):
            return {"name": values.name}
        if values is None or "name" not in values:
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
                default_context.set_value(
                    values["name"],
                    new_value,
                )

        if values.get("description") is not None:
            if not isinstance(values["description"], str):
                raise ValueError(
                    f"Description must be a string but got {type(values['description'])}."
                )
            default_context.set_metadata(values["name"], "description", values["description"])
        values.pop("description", None)

        if values.get("metadata") is not None:
            default_context.set_metadata(values["name"], "metadata", values["metadata"])
        values.pop("metadata", None)
        return values


# ---------------------------------------------------------------------------
# Client subclass: UserVariable — adds legacy variable name check
# ---------------------------------------------------------------------------

# MIGRATION-TODO(validation framework migration): Remove this subclass once
#   AllFieldNames / deprecation_reminder are available in schema.


class UserVariable(Variable, _SchemaUserVariable):
    """Client-side UserVariable with legacy variable name check."""

    @pd.field_validator("name", mode="after")
    @classmethod
    @deprecation_reminder("26.2.0")
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


# ---------------------------------------------------------------------------
# Client subclass: Expression — adds ensure_dependent_feature_enabled
# ---------------------------------------------------------------------------

# MIGRATION-TODO(validation framework migration): Remove this subclass once
#   ParamsValidationInfo + contextual_model_validator are available in schema.

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


class Expression(_SchemaExpression):
    """Client-side Expression subclass — temporary during migration.

    This subclass exists ONLY because ensure_dependent_feature_enabled requires
    ParamsValidationInfo + contextual_model_validator which have not been migrated
    to flow360-schema yet. Once the validation framework is migrated, this subclass
    should be removed and _SchemaExpression used directly everywhere.
    """

    # MIGRATION-TODO(validation framework migration): Remove this subclass entirely.
    #   Move ensure_dependent_feature_enabled to schema Expression (or lift to
    #   SimulationParams-level validation) once ParamsValidationInfo is in schema.

    @contextual_model_validator(mode="after")
    def ensure_dependent_feature_enabled(self, param_info: ParamsValidationInfo) -> str:
        """
        Ensure that all dependent features are enabled for all the solver variables.
        Remaining checks:
        1. variable valid source check.
        2. variable location check.

        """
        if self.expression not in param_info.referenced_expressions:
            return self
        # Setting recursive to False to avoid recursive error message.
        # All user variables will be checked anyways.
        for solver_variable_name in self.solver_variable_names(recursive=False):
            if solver_variable_name in _feature_requirement_map:
                if not _feature_requirement_map[solver_variable_name][0](param_info):
                    raise ValueError(
                        f"`{solver_variable_name}` cannot be used "
                        f"because {_feature_requirement_map[solver_variable_name][1]}"
                    )
        return self


# ---------------------------------------------------------------------------
# Base type aliases for isinstance checks.
# Schema operators/math functions return _SchemaExpression/_SchemaVariable instances
# (not the client subclasses above). Use these base types so isinstance checks
# match both schema and client instances.
# MIGRATION-TODO(validation framework migration): Once the client subclasses are
#   removed, these aliases become unnecessary — use Expression/Variable directly.
# ---------------------------------------------------------------------------

ExpressionBase = _SchemaExpression
VariableBase = _SchemaVariable


# ---------------------------------------------------------------------------
# Items NOT yet migrated — remain in client
# ---------------------------------------------------------------------------


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


class VariableContextInfo(Flow360BaseModel):
    """Variable context info for project variables."""

    name: str
    value: ValueOrExpression.configure(allow_run_time_expression=True)[AnyNumericType]  # type: ignore
    post_processing: bool = pd.Field()
    description: Optional[str] = pd.Field(None)
    # ** metadata is added to serve (hopefully) only front-end related purposes.
    # ** All future new keys (even if used by Python client) should be added to this field to ensure compatibility.
    metadata: Optional[dict] = pd.Field(None, description="Metadata used only by the frontend.")

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
        default_context.set_value(item.name, item.value)
    return value


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
            # Suspend unit system for legacy types; strict mode rejects bare numbers for new composed types
            with unit_system_manager.suspended(), StrictUnitContext():
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
                    if value.expression is None:
                        raise ValueError("No expression found in the input")
                    # Validate via Pydantic so that Expression validators and AfterValidator both run
                    return pd.TypeAdapter(expr_type).validate_python(
                        {"expression": value.expression, "output_units": value.output_units}
                    )

            @deprecation_reminder("26.2.0")
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
            # Use schema base classes for isinstance checks so that both schema and client
            # instances are recognized (client subclass instances also pass).
            if isinstance(v, SerializedValueOrExpression):
                return v.type_name
            if isinstance(v, dict):
                return v.get("typeName") if v.get("typeName") else v.get("type_name")
            if isinstance(v, (_SchemaExpression, _SchemaVariable, str)):
                return "expression"
            if isinstance(v, list) and all(isinstance(item, _SchemaExpression) for item in v):
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
    """
    # pylint:disable=protected-access
    post_processing_variables = get_post_processing_variables(params)
    output_units_by_name = {}
    if post_processing_variables:
        # Derive output units for all post-processing variables.
        output_units_by_name = batch_get_user_variable_units(
            list(post_processing_variables), params
        )

    user_variable_names = default_context.user_variable_names
    for name, value in list(default_context._values.items()):
        if name not in user_variable_names:
            continue

        output_unit = output_units_by_name.get(name)
        if output_unit is not None:
            output_unit_str = str(output_unit)
            if isinstance(value, ExpressionBase):
                value = value.model_copy()
                value.output_units = output_unit_str
            else:
                value = VariableContextInfo.convert_number_to_expression(value)
                value.output_units = output_unit_str

        if params.private_attribute_asset_cache.variable_context is None:
            params.private_attribute_asset_cache.variable_context = []

        # Check if variable with this name already exists
        existing_index = None
        for i, existing_var in enumerate(params.private_attribute_asset_cache.variable_context):
            if existing_var.name == name:
                existing_index = i
                break

        new_variable = VariableContextInfo(
            name=name,
            value=value,
            description=default_context.get_metadata(name, "description"),
            post_processing=name in post_processing_variables,
            metadata=default_context.get_metadata(name, "metadata"),
        )

        if existing_index is not None:
            # Replace existing variable
            params.private_attribute_asset_cache.variable_context[existing_index] = new_variable
        else:
            # Append new variable
            params.private_attribute_asset_cache.variable_context.append(new_variable)
    return params


def batch_get_user_variable_units(variable_names: list[str], params):
    """
    Return output units for a list of user variable names.

    For each name, the value is pulled from `default_context` and converted to a unit:
    - Expression: `Expression.get_output_units(params)` (respects explicit output_units or
      infers from `params.unit_system`).
    - unyt_array/unyt_quantity: their `units`.
    - Number: "dimensionless".

    Returns a dict mapping variable name to a `unyt.Unit` (or the string "dimensionless").
    Raises `ValueError` if a name resolves to an unsupported type.
    """
    result = {}
    for name in variable_names:
        value = default_context.get(name)
        if isinstance(value, ExpressionBase):
            result[name] = value.get_output_units(params)
        elif isinstance(value, unyt_array):
            result[name] = value.units
        elif isinstance(value, Number):
            result[name] = "dimensionless"
        else:
            raise ValueError(f"Unknown variable type: {value}")
    return result


def get_referenced_expressions_and_user_variables(param_as_dict: dict):
    """
    Get all the expressions that are mentioned/referenced in the params dict
    (excluding the ones that are in the asset cache)
    Two sources:
    1. Field is `Expression`.
    2. Field is `UserVariable` and `value` is an `Expression`.
    `Expression` and `UserVariable` are both identified by their schema.
    """

    def _is_user_variable(field: dict) -> bool:
        return "type_name" in field and field["type_name"] == "UserVariable"

    def _is_expression(field: dict) -> bool:
        if "type_name" in field and field["type_name"] == "expression":
            return True
        if sorted(field.keys()) == ["expression", "output_units"] or sorted(field.keys()) == [
            "expression"
        ]:
            return True
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
                if "." not in var.name and isinstance(var.value, ExpressionBase):
                    dependent_expressions.add(str(var.value))
                    _get_dependent_expressions(var.value, dependent_expressions)
            except ValueError:
                # An undefined variable is found. Validation will handle this.
                pass

    def _collect_expressions_recursive(
        data,
        used_expressions: set,
        current_path: tuple[str, ...] = (),
        exclude_paths: set[tuple[str, ...]] = (
            ("private_attribute_asset_cache", "variable_context"),
        ),
    ):
        # pylint: disable=too-many-branches
        """
        Recursively collect expressions from nested data structures.

        current_path tracks the traversal keys from the root. If current_path matches
        any tuple in exclude_paths, the sub-tree is skipped. seen_ids prevents revisiting
        the same object multiple times when shared references exist.
        """
        if data is None or isinstance(data, (int, float, str, bool)):
            return

        if current_path in exclude_paths:
            return

        if isinstance(data, dict):
            # Check if this dict is a UserVariable
            if _is_user_variable(data):
                variable_name = data.get("name", {})
                if "." in variable_name:
                    return
                try:
                    value = default_context.get(variable_name)
                    if isinstance(value, ExpressionBase):
                        used_expressions.add(str(value))
                except ValueError:
                    # An undefined variable is found. Validation will handle this.
                    pass

            # Check if this dict is an Expression
            elif _is_expression(data):
                used_expressions.add(data.get("expression"))

            # Recursively process all values in the dict
            for key, value in data.items():
                _collect_expressions_recursive(
                    value,
                    used_expressions,
                    current_path + (key,),
                    exclude_paths,
                )

        elif isinstance(data, list):
            # Recursively process all items in the list
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
        return [], []

    used_expressions: set[str] = set()
    _collect_expressions_recursive(
        param_as_dict,
        used_expressions,
        current_path=(),
        exclude_paths={("private_attribute_asset_cache", "variable_context")},
    )

    dependent_expressions = set()

    for expr in used_expressions:
        _get_dependent_expressions(Expression(expression=expr), dependent_expressions)

    return list(used_expressions.union(dependent_expressions))


def solver_variable_to_user_variable(item):
    """Convert the solver variable to a user variable using the current unit system."""
    if isinstance(item, SolverVariable):
        if unit_system_manager.current is None:
            raise ValueError(f"Solver variable {item.name} cannot be used without a unit system.")
        unit_system_name = unit_system_manager.current.name
        name = item.name.split(".")[-1] if "." in item.name else item.name
        return UserVariable(name=f"{name}_{unit_system_name}", value=item)
    return item


def is_variable_with_unit_system_as_units(value: dict) -> bool:
    """
    [Frontend] Check if the value is a variable with a unit system as units.
    """
    return (
        not isinstance(value, dict)
        or "units" not in value
        or value["units"]
        not in (
            "SI_unit_system",
            "Imperial_unit_system",
            "CGS_unit_system",
        )
    )


def infer_units_by_unit_system(value: dict, unit_system: str, value_dimensions):
    """
    [Frontend] Infer the units based on the unit system.
    """
    if unit_system == "SI_unit_system":
        value["units"] = u.unit_systems.mks_unit_system[value_dimensions]
    if unit_system == "Imperial_unit_system":
        value["units"] = u.unit_systems.imperial_unit_system[value_dimensions]
    if unit_system == "CGS_unit_system":
        value["units"] = u.unit_systems.cgs_unit_system[value_dimensions]
    return value


def compute_surface_integral_unit(variable: UserVariable, params) -> str:
    """
    Compute the unit of the surface integral of a UserVariable over a surface.
    """
    base_unit = None
    if isinstance(variable.value, ExpressionBase):
        base_unit = variable.value.get_output_units(params)
    else:
        val = variable.value
        if hasattr(val, "get_output_units"):
            base_unit = val.get_output_units(params)
        elif isinstance(val, (unyt_array, unyt_quantity)):
            base_unit = val.units
        elif isinstance(val, Number):
            base_unit = u.Unit("dimensionless")
        else:
            base_unit = u.Unit("dimensionless")

    if base_unit is None:
        # Fallback if output_units is not set for expression or if it is a number
        base_unit = u.Unit("dimensionless")

    area_unit = params.unit_system.resolve()["area"].units
    result_unit = base_unit * area_unit
    return str(result_unit)
