"""Serialized variable-context payload models used outside the expression runtime package."""

from typing import Any

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel


class VariableContextInfo(Flow360BaseModel):
    """Serialized project-variable entry stored in AssetCache.

    The ``value`` field starts as ``Any`` so schema import/export of ``AssetCache``
    does not pull in the full expression runtime dependency graph. The expression
    package patches this field to the real ``ValueOrExpression`` type at runtime.
    """

    name: str
    value: Any
    post_processing: bool = pd.Field()
    description: str | None = pd.Field(None)
    metadata: dict[str, Any] | None = pd.Field(None, description="Metadata used only by the frontend.")

    @pd.field_validator("value", mode="after")
    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Preserve runtime VariableContextInfo semantics after model unification."""
        from flow360_schema.framework.expression.variable import _convert_variable_value_to_expression

        return _convert_variable_value_to_expression(value)


VariableContextList = list[VariableContextInfo]
