"""Client-only adapters for the expression system.

Core types (Variable, UserVariable, SolverVariable, Expression, VariableContextInfo) live in
flow360-schema. This file provides ValueOrExpression (client subclass) and adapter functions
that depend on client-specific state (unit_system_manager, deprecation_reminder, params).
"""

from __future__ import annotations

from numbers import Number
from typing import Annotated, Any, ClassVar, Generic, List, TypeVar, Union

import numpy as np
import pydantic as pd
import unyt as u
from flow360_schema import StrictUnitContext
from flow360_schema.framework.expression.utils import is_runtime_expression
from flow360_schema.framework.expression.value_or_expression import (
    SerializedValueOrExpression,
    register_deprecation_check,
)
from flow360_schema.framework.expression.variable import (
    Expression,
    UserVariable,
    Variable,
    _check_list_items_are_same_dimensions,
)
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from typing_extensions import Self
from unyt import unyt_array, unyt_quantity

from flow360.component.simulation.framework.updater_utils import deprecation_reminder
from flow360.component.simulation.unit_system import unit_system_manager

register_deprecation_check(deprecation_reminder)

T = TypeVar("T")


# TODO(migration): Migrate to schema once deprecation_reminder is migrated.
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


# TODO(migration): Migrate to schema once params.outputs structure is available in schema.
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


# TODO(migration): Migrate to schema once get_post_processing_variables and
# params.private_attribute_asset_cache are available in schema.
def save_user_variables(params):
    """Client adapter: extract data from params, delegate to schema."""
    # pylint:disable = import-outside-toplevel
    from flow360_schema.framework.expression.variable import (
        batch_get_user_variable_units as _schema_batch_get_user_variable_units,
    )
    from flow360_schema.framework.expression.variable import (
        save_user_variables as _schema_save_user_variables,
    )

    post_processing_variables = get_post_processing_variables(params)
    output_units = {}
    if post_processing_variables:
        output_units = {
            name: str(unit)
            for name, unit in _schema_batch_get_user_variable_units(
                list(post_processing_variables), params.unit_system.name
            ).items()
        }

    result = _schema_save_user_variables(
        variable_context=params.private_attribute_asset_cache.variable_context,
        post_processing_names=post_processing_variables,
        output_units=output_units,
    )
    params.private_attribute_asset_cache.variable_context = result
    return params


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
