"""
Module for validation context handling in the simulation component of Flow360.

This module provides context management for validation levels and specialized field
definitions for conditional validation scenarios.

Features
--------
- This module allows for defining conditionally (context-based) required fields,
  such as fields required only for specific scenarios like surface mesh or volume mesh.
- It supports running validation only for specific scenarios (surface mesh, volume mesh, or case),
  allowing for targeted validation flows based on the current context.
- This module does NOT ignore validation errors; instead, it enriches errors with context
  information, enabling downstream processes to filter and interpret errors based on scenario-specific requirements.
"""

import contextvars
from functools import wraps
from typing import Any, Callable, List, Literal, Union

from pydantic import Field

SURFACE_MESH = "SurfaceMesh"
VOLUME_MESH = "VolumeMesh"
CASE = "Case"
# when running validation with ALL, it will report errors happing in all scenarios in one validation pass
ALL = "All"

_validation_level_ctx = contextvars.ContextVar("validation_levels", default=None)


class ValidationLevelContext:
    """
    Context manager for setting the validation level.

    Allows setting a specific validation level within a context, which influences
    the conditional validation of fields based on the defined levels.
    """

    def __init__(self, levels: Union[str, List[str]]):
        valid_levels = {SURFACE_MESH, VOLUME_MESH, CASE, ALL}
        if isinstance(levels, str):
            levels = [levels]
        if (
            levels is None
            or isinstance(levels, list)
            and all(lvl in valid_levels for lvl in levels)
        ):
            self.levels = levels
            self.token = None
        else:
            raise ValueError(f"Invalid validation level: {levels}")

    def __enter__(self):
        self.token = _validation_level_ctx.set(self.levels)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _validation_level_ctx.reset(self.token)


def get_validation_levels() -> list:
    """
    Retrieves the current validation level from the context.

    Returns:
        The current validation level, which can influence field validation behavior.
    """
    return _validation_level_ctx.get()


# pylint: disable=invalid-name
def ContextField(
    default=None, *, context: Literal["SurfaceMesh", "VolumeMesh", "Case"] = None, **kwargs
):
    """
    Creates a field with context validation (the field is not required).

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    context : str, optional
        Specifies the scenario for which this field is relevant. The relevant_for=context is included in error->ctx
    **kwargs : dict
        Additional keyword arguments passed to Pydantic's Field.

    Returns
    -------
    Field
        A Pydantic Field configured with conditional validation context.

    Notes
    -----
    Use this field for not required fields to profide context information during validation.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": False},
        **kwargs,
    )


# pylint: disable=invalid-name
def ConditionalField(
    default=None,
    *,
    context: Union[
        None,
        Literal["SurfaceMesh", "VolumeMesh", "Case"],
        List[Literal["SurfaceMesh", "VolumeMesh", "Case"]],
    ] = None,
    **kwargs,
):
    """
    Creates a field with conditional context validation requirements.

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    context : Union[
        None, Literal['SurfaceMesh', 'VolumeMesh', 'Case'],
        List[Literal['SurfaceMesh', 'VolumeMesh', 'Case']]
    ], optional
        Specifies the context(s) in which this field is relevant. This value is
        included in the validation error context (`ctx`) as `relevant_for`.
    **kwargs : dict
        Additional keyword arguments passed to Pydantic's Field.

    Returns
    -------
    Field
        A Pydantic Field configured with conditional validation context.

    Notes
    -----
    Use this field for required fields but only for certain scenarios, such as volume meshing only.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": True},
        **kwargs,
    )


# pylint: disable=invalid-name
def CaseField(default=None, **kwargs):
    """
    Creates a field specifically relevant for the Case scenario.

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    **kwargs : dict
        Additional keyword arguments passed to the pd.Field().

    Returns
    -------
    Field
        A Pydantic Field configured for fields relevant only to the Case scenario.

    Notes
    -----
    Use this field for fields that are not required but make sense only for Case.
    """
    return ContextField(default, context=CASE, **kwargs)


def context_validator(context: Literal["SurfaceMesh", "VolumeMesh", "Case"]):
    """
    Decorator to conditionally run a validator based on the current validation context.

    This decorator runs the decorated validator function only if the current validation
    level matches the specified context or if the validation level is set to ALL.

    Parameters
    ----------
    context : Literal["SurfaceMesh", "VolumeMesh", "Case"]
        The specific validation context in which the validator should be run.

    Returns
    -------
    Callable
        The decorated function that will only run when the specified context condition is met.

    Notes
    -----
    This decorator is designed to be used with Pydantic model validators to ensure that
    certain validations are only executed when the validation level matches the given context.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self: Any, *args, **kwargs):
            current_levels = get_validation_levels()
            # Run the validator only if the current levels matches the specified context or is ALL
            if current_levels is None or any(lvl in (context, ALL) for lvl in current_levels):
                return func(self, *args, **kwargs)
            return self

        return wrapper

    return decorator
