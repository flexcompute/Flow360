"""ValueOrExpression: generic discriminated union accepting both numeric values and expressions.

Migrated from Flow360 Python client (user_code/core/types.py).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from numbers import Number
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
)

import numpy as np
import pydantic as pd
import unyt as u
from pydantic import BeforeValidator, Discriminator, PlainSerializer, Tag
from pydantic_core import core_schema
from unyt import unyt_array, unyt_quantity

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.expression.utils import is_runtime_expression
from flow360_schema.framework.expression.variable import (
    Expression,
    Variable,
    _check_list_items_are_same_dimensions,
)
from flow360_schema.framework.validation.context import StrictUnitContext, unit_system_manager

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Deprecation check hook — injectable from client side
# ---------------------------------------------------------------------------
# Schema cannot import client's deprecation_reminder. Client registers it via
# register_deprecation_check() at import time.  When set, the hook is called as
# ``_deprecation_check(version)(func)`` and returns the decorated function.
# MIGRATION-TODO: Once updaters (updater_utils.py, deprecation_reminder, Flow360Version) are
# migrated to schema, import deprecation_reminder directly and remove this callback mechanism.
_deprecation_check: Callable[[str], Callable[..., Any]] | None = None


def register_deprecation_check(checker: Callable[[str], Callable[..., Any]]) -> None:
    """Register a deprecation-check decorator factory (e.g. client's deprecation_reminder)."""
    global _deprecation_check  # noqa: PLW0603
    _deprecation_check = checker


# ---------------------------------------------------------------------------
# Serialized envelope
# ---------------------------------------------------------------------------


class SerializedValueOrExpression(Flow360BaseModel):
    """Serialized frontend-compatible format of an arbitrary value/expression field"""

    type_name: Literal["number", "expression"] = pd.Field()
    value: Number | list[Number] | None = pd.Field(None)
    units: str | None = pd.Field(None)
    expression: str | None = pd.Field(None)
    output_units: str | None = pd.Field(None, description="See definition in `Expression`.")


# ---------------------------------------------------------------------------
# Pydantic-compatible unyt wrappers
# ---------------------------------------------------------------------------


class UnytQuantity(unyt_quantity):  # type: ignore[misc]
    """UnytQuantity wrapper to enable pydantic compatibility"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any) -> unyt_quantity:
        """Minimal validator for pydantic compatibility"""
        if isinstance(value, unyt_quantity):
            return value
        if isinstance(value, unyt_array) and value.shape == ():
            # When deserialized unyt_quantity() gives unyt_array
            return unyt_quantity(value.value, value.units)
        raise ValueError("Input should be a valid unit quantity.")


class UnytArray(unyt_array):  # type: ignore[misc]
    """UnytArray wrapper to enable pydantic compatibility"""

    def __repr__(self) -> str:
        return f"UnytArray({str(self)})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any) -> unyt_array:
        """Minimal validator for pydantic compatibility"""
        if isinstance(value, unyt_array):
            return value
        raise ValueError(f"Cannot convert {type(value)} to UnytArray")


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

AnyNumericType = float | UnytArray | list[Any]


# ---------------------------------------------------------------------------
# ValueOrExpression
# ---------------------------------------------------------------------------


class ValueOrExpression(Expression, Generic[T]):
    """Model accepting both value and expressions"""

    _cfg: ClassVar[dict[str, Any]] = {}

    @classmethod
    def configure(cls, **flags: Any) -> type[ValueOrExpression[Any]]:
        """
        Create a new subclass with the given flags.
        """
        name = f"{cls.__name__}[{','.join(f'{k}={v}' for k, v in flags.items())}]"
        return type(name, (cls,), {"_cfg": {**cls._cfg, **flags}})

    def __class_getitem__(cls, typevar_values: Any) -> Any:
        cfg = cls._cfg
        # By default all value or expression should be able to be evaluated at compile-time
        allow_run_time_expression = bool(cfg.get("allow_run_time_expression", False))

        def _internal_validator(value: Expression) -> Expression:
            try:
                # Symbolically validate
                value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
                # Numerically validate
                result = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            except Exception as err:
                raise ValueError(f"expression evaluation failed: {err}") from err

            # Detect run-time expressions
            if not allow_run_time_expression and is_runtime_expression(result):
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

        def _deserialize(value: Any) -> Any:
            # Try to see if the value is already a SerializedValueOrExpression
            try:  # noqa: SIM105
                value = SerializedValueOrExpression.model_validate(value)
            except Exception:
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

            # Handle list of unyt_quantities:
            if isinstance(value, list):
                if len(value) == 0:
                    raise ValueError("Empty list is not allowed.")
                _check_list_items_are_same_dimensions(value)
                if all(isinstance(item, (unyt_quantity, Number)) for item in value):
                    # try limiting the number of types we need to handle
                    return unyt_array(value, dtype=np.float64)
            return value

        def _serializer(value: Any, info: Any) -> dict[str, Any]:
            if isinstance(value, Expression):
                serialized = SerializedValueOrExpression(  # type: ignore[call-arg]
                    type_name="expression",
                    output_units=value.output_units,
                )

                serialized.expression = value.expression

                evaluated = value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)

                if isinstance(evaluated, list):
                    # May result from Expression which is actually a list of expressions
                    with contextlib.suppress(u.exceptions.IterableUnitCoercionError):
                        evaluated = u.unyt_array(evaluated, dtype=np.float64)
            else:
                serialized = SerializedValueOrExpression(type_name="number")  # type: ignore[call-arg]
                # Note: NaN handling should be unnecessary since it would
                # have end up being expression first so not reaching here.
                if isinstance(value, (Number, list)):
                    serialized.value = value
                elif isinstance(value, u.unyt_array):
                    if value.size == 1:
                        serialized.value = float(value.value)  # type: ignore[assignment]
                    else:
                        serialized.value = tuple(value.value.tolist())  # type: ignore[assignment]

                    serialized.units = str(value.units.expr)

            return serialized.model_dump(**info.__dict__)

        def _discriminator(v: Any) -> str:
            # Note: This is ran after deserializer
            # Use schema base classes for isinstance checks so that both schema and client
            # instances are recognized (client subclass instances also pass).
            if isinstance(v, SerializedValueOrExpression):
                return v.type_name
            if isinstance(v, dict):
                type_name = v.get("typeName") or v.get("type_name")
                if type_name is not None:
                    return type_name  # type: ignore[no-any-return]
                # The updater migrates legacy `{units, value}` to
                # `{value, [display_unit]}` before deserialization runs. Route
                # this dict form to the number branch so the typevar's composed-
                # type validator parses it via the display-unit-dict path.
                if "value" in v and "units" not in v:
                    return "number"
                return None  # type: ignore[return-value]
            if isinstance(v, (Expression, Variable, str)):
                return "expression"
            if isinstance(v, list) and all(isinstance(item, Expression) for item in v):
                return "expression"
            if isinstance(v, (Number, unyt_array, list)):
                return "number"
            raise KeyError("Unknown expression input type: ", v, v.__class__.__name__)

        union_type = Annotated[
            Annotated[expr_type, Tag("expression")] | Annotated[typevar_values, Tag("number")],
            pd.Field(discriminator=Discriminator(_discriminator)),
            BeforeValidator(_deserialize),
            PlainSerializer(_serializer),
        ]
        return union_type


__all__ = [
    "SerializedValueOrExpression",
    "ValueOrExpression",
    "UnytQuantity",
    "UnytArray",
    "AnyNumericType",
    "register_deprecation_check",
]
