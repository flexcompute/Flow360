"""Client adapters for expression types and params-dependent helpers."""

from __future__ import annotations

from flow360_schema.framework.expression.value_or_expression import (
    register_deprecation_check,
)
from flow360_schema.framework.expression.variable import (
    batch_get_user_variable_units as _schema_batch_get_user_variable_units,
)
from flow360_schema.framework.expression.variable import (
    save_user_variables as _schema_save_user_variables,
)
from flow360_schema.models.simulation.framework.updater_utils import (
    deprecation_reminder,
)
from flow360_schema.models.simulation.user_code.core.types import (
    get_post_processing_variables,
)

register_deprecation_check(deprecation_reminder)


def save_user_variables(params):
    """Client adapter: extract data from params, delegate to schema."""
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


__all__ = [
    "save_user_variables",
]
