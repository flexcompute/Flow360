"""Scoped context for entity materialization and reuse.

This module provides a context-managed cache and an injectable builder
for converting entity dictionaries to model instances, avoiding global
state and enabling high-performance reuse during validation.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Optional

_entity_cache_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "entity_cache", default=None
)
_entity_builder_ctx: contextvars.ContextVar[Optional[Callable[[dict], Any]]] = (
    contextvars.ContextVar("entity_builder", default=None)
)


class EntityMaterializationContext:
    """Context manager providing a per-validation scoped cache and builder.

    Use this to avoid global state when materializing entity dictionaries
    into model instances while reusing objects across the validation pass.

    Parameters
    ----------
    builder : Callable[[dict], Any]
        Function to convert entity dict to instance when not found in cache.
    entity_pool : Optional[dict]
        Pre-existing entity instances keyed by (type_name, private_attribute_id).
        When provided, entities matching these keys will reuse the pool instances
        instead of creating new ones via builder.
    """

    def __init__(self, *, builder: Callable[[dict], Any], entity_pool: Optional[dict] = None):
        self._token_cache = None
        self._token_builder = None
        self._builder = builder
        self._entity_pool = entity_pool

    def __enter__(self):
        # Pre-populate cache from entity_pool if provided
        initial_cache = {}
        if self._entity_pool:
            initial_cache = dict(self._entity_pool)  # Copy to avoid external mutation
        self._token_cache = _entity_cache_ctx.set(initial_cache)
        self._token_builder = _entity_builder_ctx.set(self._builder)
        return self

    def __exit__(self, exc_type, exc, tb):
        _entity_cache_ctx.reset(self._token_cache)
        _entity_builder_ctx.reset(self._token_builder)


def get_entity_cache() -> Optional[dict]:
    """Return the current cache dict for entity reuse, or None if not active."""

    return _entity_cache_ctx.get()


def get_entity_builder() -> Optional[Callable[[dict], Any]]:
    """Return the current dict->entity builder, or None if not active."""

    return _entity_builder_ctx.get()
