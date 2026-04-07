"""Client adapters for expression types and params-dependent helpers."""

from __future__ import annotations

import unyt as u
from flow360_schema.framework.expression import (
    SerializedValueOrExpression,
    UnytQuantity,
    UserVariable,
    ValueOrExpression,
)
from flow360_schema.framework.expression.value_or_expression import (
    register_deprecation_check,
)

from flow360.component.simulation.framework.updater_utils import deprecation_reminder

register_deprecation_check(deprecation_reminder)


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
        if "output_fields" not in item.__class__.model_fields:
            continue
        for output_field in item.output_fields.items:
            if isinstance(output_field, UserVariable):
                post_processing_variables.add(output_field.name)
    return post_processing_variables


def save_user_variables(params):
    """Client adapter: extract data from params, delegate to schema."""
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


__all__ = [
    "SerializedValueOrExpression",
    "UnytQuantity",
    "ValueOrExpression",
    "get_post_processing_variables",
    "infer_units_by_unit_system",
    "is_variable_with_unit_system_as_units",
    "save_user_variables",
]
