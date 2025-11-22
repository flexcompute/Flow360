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
    """

    def __init__(self, *, builder: Callable[[dict], Any]):
        self._token_cache = None
        self._token_builder = None
        self._builder = builder

    def __enter__(self):
        self._token_cache = _entity_cache_ctx.set({})
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
