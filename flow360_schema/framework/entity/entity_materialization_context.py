"""Scoped context for entity materialization and reuse.

This module provides a context-managed cache and an injectable builder
for converting entity dictionaries to model instances, avoiding global
state and enabling high-performance reuse during validation.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_registry import EntityRegistry

_entity_cache_ctx: contextvars.ContextVar[dict[Any, Any] | None] = contextvars.ContextVar("entity_cache", default=None)
_entity_builder_ctx: contextvars.ContextVar[Callable[[dict[str, Any]], Any] | None] = contextvars.ContextVar(
    "entity_builder", default=None
)
_entity_registry_ctx: contextvars.ContextVar[EntityRegistry | None] = contextvars.ContextVar(
    "entity_registry", default=None
)


class EntityMaterializationContext:
    """Context manager providing a per-validation scoped cache and builder.

    Use this to avoid global state when materializing entity dictionaries
    into model instances while reusing objects across the validation pass.

    Parameters
    ----------
    builder : Callable[[dict], Any]
        Function to convert entity dict to instance when not found in cache.
    entity_registry : Optional[EntityRegistry]
        Pre-existing EntityRegistry containing canonical entity instances.
        When provided, entities are looked up by (type_name, private_attribute_id)
        and must exist in the registry (errors if not found).
    """

    def __init__(
        self,
        *,
        builder: Callable[[dict[str, Any]], Any],
        entity_registry: EntityRegistry | None = None,
    ) -> None:
        self._token_cache: contextvars.Token[dict[Any, Any] | None] | None = None
        self._token_builder: contextvars.Token[Callable[[dict[str, Any]], Any] | None] | None = None
        self._supplied_entity_registry: contextvars.Token[EntityRegistry | None] | None = None
        self._builder = builder
        self._entity_registry = entity_registry

    def __enter__(self) -> EntityMaterializationContext:
        # Set up context variables
        initial_cache: dict[Any, Any] = {}
        self._token_cache = _entity_cache_ctx.set(initial_cache)
        self._token_builder = _entity_builder_ctx.set(self._builder)
        self._supplied_entity_registry = _entity_registry_ctx.set(self._entity_registry)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        _entity_cache_ctx.reset(self._token_cache)  # type: ignore[arg-type]
        _entity_builder_ctx.reset(self._token_builder)  # type: ignore[arg-type]
        _entity_registry_ctx.reset(self._supplied_entity_registry)  # type: ignore[arg-type]


def get_entity_cache() -> dict[Any, Any] | None:
    """Return the current cache dict for entity reuse, or None if not active."""

    return _entity_cache_ctx.get()


def get_entity_builder() -> Callable[[dict[str, Any]], Any] | None:
    """Return the current dict->entity builder, or None if not active."""

    return _entity_builder_ctx.get()


def get_entity_registry() -> EntityRegistry | None:
    """Return the current EntityRegistry for entity lookup, or None if not active."""

    return _entity_registry_ctx.get()
