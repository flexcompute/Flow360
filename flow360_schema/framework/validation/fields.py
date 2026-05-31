"""Conditional and context-aware field factories for Flow360 schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .context import CASE


def ContextField(
    default: Any = None,
    *,
    context: Literal["SurfaceMesh", "VolumeMesh", "Case"] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a field with context validation (the field is not required).

    Use this for optional fields to provide context information during validation.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": False},
        **kwargs,
    )


def ConditionalField(
    default: Any = None,
    *,
    context: (
        None | Literal["SurfaceMesh", "VolumeMesh", "Case"] | list[Literal["SurfaceMesh", "VolumeMesh", "Case"]]
    ) = None,
    **kwargs: Any,
) -> Any:
    """Create a field with conditional context validation requirements.

    Use this for fields required only in certain scenarios (e.g. volume meshing only).
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": True},  # type: ignore[dict-item]  # Pydantic accepts flexible json_schema_extra types
        **kwargs,
    )


def CaseField(default: Any = None, **kwargs: Any) -> Any:
    """Create a field specifically relevant for the Case scenario."""
    return ContextField(default, context=CASE, **kwargs)  # type: ignore[arg-type]  # CASE is str constant that matches Literal type at runtime
