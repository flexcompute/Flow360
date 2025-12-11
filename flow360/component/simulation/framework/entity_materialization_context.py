"""Scoped context for entity materialization and reuse.

This module provides a context-managed cache and an injectable builder
for converting entity dictionaries to model instances, avoiding global
state and enabling high-performance reuse during validation.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from flow360.component.simulation.framework.entity_registry import EntityRegistry

_entity_cache_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "entity_cache", default=None
)
_entity_builder_ctx: contextvars.ContextVar[Optional[Callable[[dict], Any]]] = (
    contextvars.ContextVar("entity_builder", default=None)
)
_entity_registry_ctx: contextvars.ContextVar[Optional[EntityRegistry]] = contextvars.ContextVar(
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
        builder: Callable[[dict], Any],
        entity_registry: Optional[EntityRegistry] = None,
    ):
        self._token_cache = None
        self._token_builder = None
        self._supplied_entity_registry = None
        self._builder = builder
        self._entity_registry = entity_registry

    def __enter__(self):
        # Set up context variables
        initial_cache = {}
        self._token_cache = _entity_cache_ctx.set(initial_cache)
        self._token_builder = _entity_builder_ctx.set(self._builder)
        self._supplied_entity_registry = _entity_registry_ctx.set(self._entity_registry)
        return self

    def __exit__(self, exc_type, exc, tb):
        _entity_cache_ctx.reset(self._token_cache)
        _entity_builder_ctx.reset(self._token_builder)
        _entity_registry_ctx.reset(self._supplied_entity_registry)


def get_entity_cache() -> Optional[dict]:
    """Return the current cache dict for entity reuse, or None if not active."""

    return _entity_cache_ctx.get()


def get_entity_builder() -> Optional[Callable[[dict], Any]]:
    """Return the current dict->entity builder, or None if not active."""

    return _entity_builder_ctx.get()


def get_entity_registry() -> Optional[EntityRegistry]:
    """Return the current EntityRegistry for entity lookup, or None if not active."""

    return _entity_registry_ctx.get()
