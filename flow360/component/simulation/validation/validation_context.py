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
from typing import Optional

from pydantic import Field

SURFACE_MESH = "SurfaceMesh"
VOLUME_MESH = "VolumeMesh"
CASE = "Case"
# when running validation with ALL, it will report errors happing in all scenarios in one validation pass
ALL = "All"

_validation_level_ctx = contextvars.ContextVar("validation_level", default=None)


class ValidationLevelContext:
    """
    Context manager for setting the validation level.

    Allows setting a specific validation level within a context, which influences
    the conditional validation of fields based on the defined levels.
    """

    def __init__(self, level: str):
        if level not in {None, SURFACE_MESH, VOLUME_MESH, CASE, ALL}:
            raise ValueError(f"Invalid validation level: {level}")
        self.level = level
        self.token = None

    def __enter__(self):
        self.token = _validation_level_ctx.set(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _validation_level_ctx.reset(self.token)


def get_validation_level():
    """
    Retrieves the current validation level from the context.

    Returns:
        The current validation level, which can influence field validation behavior.
    """
    return _validation_level_ctx.get()


# pylint: disable=invalid-name
def ConditionalField(
    default=None, *, relevant_for: Optional[str] = None, required: Optional[bool] = False, **kwargs
):
    """
    Creates a field with conditional validation requirements.

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    relevant_for : str, optional
        Specifies the scenario for which this field is relevant. The relevant_for is included in error->ctx
    required : bool, optional
        Indicates if the field is conditionally required based on the scenario.
    **kwargs : dict
        Additional keyword arguments passed to Pydantic's Field.

    Returns
    -------
    Field
        A Pydantic Field configured with conditional validation context.

    Notes
    -----
    Use this field for required or not required fields but only for certain scenarios,
    such as volume meshing only.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": relevant_for, "conditionally_required": required},
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
    return ConditionalField(default, relevant_for=CASE, **kwargs)
