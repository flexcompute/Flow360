"""Validation context management for conditional field validation.

Provides context variables and a context manager for controlling which
validation levels (SurfaceMesh, VolumeMesh, Case) are active during
model construction.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator
from typing import Any

SURFACE_MESH = "SurfaceMesh"
VOLUME_MESH = "VolumeMesh"
CASE = "Case"
# When running validation with ALL, it will report errors for all scenarios in one pass
ALL = "All"

_validation_level_ctx: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar(
    "validation_levels", default=None
)
_validation_info_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("validation_info", default=None)
_validation_warnings_ctx: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar(
    "validation_warnings", default=None
)

_deserializing_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("deserializing", default=False)
_strict_unit_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("strict_unit", default=False)
_active_unit_system_ctx: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "active_unit_system",
    default=None,
)


class DeserializationContext:
    """Context manager to mark a scope as deserialization (suppresses bare-numeric warnings)."""

    def __enter__(self) -> DeserializationContext:
        self._token = _deserializing_ctx.set(True)
        return self

    def __exit__(self, *_: Any) -> None:
        _deserializing_ctx.reset(self._token)


def is_deserializing() -> bool:
    """Check if currently in a deserialization scope."""
    return _deserializing_ctx.get()


class StrictUnitContext:
    """Context manager that rejects bare numbers without explicit units.

    When active, bare numeric values (without unit) will raise ValueError
    instead of falling back to SI. Used by Expression validation to enforce
    that evaluated results carry explicit units.
    """

    def __enter__(self) -> StrictUnitContext:
        self._token = _strict_unit_ctx.set(True)
        return self

    def __exit__(self, *_: Any) -> None:
        _strict_unit_ctx.reset(self._token)


def is_strict_unit_mode() -> bool:
    """Check if currently in a strict unit scope (bare numbers rejected)."""
    return _strict_unit_ctx.get()


_suspend_tokens_ctx: contextvars.ContextVar[list[contextvars.Token[Any | None]]] = contextvars.ContextVar(
    "suspend_tokens"
)


class UnitSystemManager:
    """Manage active unit system state via ContextVar-backed context tokens."""

    @property
    def current(self) -> Any | None:
        """Get current active unit system from context."""
        return _active_unit_system_ctx.get()

    def set_current(self, unit_system: Any | None) -> contextvars.Token[Any | None]:
        """Set active unit system and return token for scoped reset."""
        return _active_unit_system_ctx.set(unit_system)

    def reset_current(self, token: contextvars.Token[Any | None]) -> None:
        """Reset active unit system by token."""
        _active_unit_system_ctx.reset(token)

    def _get_suspend_tokens(self) -> list[contextvars.Token[Any | None]]:
        try:
            return _suspend_tokens_ctx.get()
        except LookupError:
            tokens: list[contextvars.Token[Any | None]] = []
            _suspend_tokens_ctx.set(tokens)
            return tokens

    def suspend(self) -> None:
        """Temporarily clear active unit system and remember token for resume."""
        tokens = self._get_suspend_tokens()
        _suspend_tokens_ctx.set([*tokens, self.set_current(None)])

    def resume(self) -> None:
        """Restore last suspended active unit system."""
        tokens = self._get_suspend_tokens()
        if not tokens:
            raise RuntimeError("No suspended unit system context to resume.")
        self.reset_current(tokens[-1])
        _suspend_tokens_ctx.set(tokens[:-1])

    @contextlib.contextmanager
    def suspended(self) -> Generator[None, None, None]:
        """Context manager that suspends and guarantees resume on exit."""
        self.suspend()
        try:
            yield
        finally:
            self.resume()


unit_system_manager = UnitSystemManager()


class ValidationContext:
    """Context manager for setting validation level and additional background.

    Allows setting a specific validation level within a context, which influences
    the conditional validation of fields based on the defined levels.
    """

    def __init__(self, levels: str | list[str], info: Any = None) -> None:
        valid_levels = {SURFACE_MESH, VOLUME_MESH, CASE, ALL}
        if isinstance(levels, str):
            levels = [levels]
        if levels is None or (isinstance(levels, list) and all(lvl in valid_levels for lvl in levels)):
            self.levels = levels
            self.level_token: contextvars.Token[list[Any] | None] | None = None
            self.validation_warnings: list[Any] = []
        else:
            raise ValueError(f"Invalid validation level: {levels}")

        self.info = info
        self.info_token: contextvars.Token[Any] | None = None
        self.warnings_token: contextvars.Token[list[Any] | None] | None = None

    def __enter__(self) -> ValidationContext:
        self.level_token = _validation_level_ctx.set(self.levels)
        self.info_token = _validation_info_ctx.set(self.info)
        self.warnings_token = _validation_warnings_ctx.set(self.validation_warnings)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.level_token is not None:
            _validation_level_ctx.reset(self.level_token)
        if self.info_token is not None:
            _validation_info_ctx.reset(self.info_token)
        if self.warnings_token is not None:
            _validation_warnings_ctx.reset(self.warnings_token)


def get_validation_levels() -> list[Any] | None:
    """Retrieve the current validation levels from context."""
    return _validation_level_ctx.get()


def get_validation_info() -> Any:
    """Retrieve the current validation info from context."""
    return _validation_info_ctx.get()


def add_validation_warning(message: str) -> None:
    """Append a validation warning message to the active ValidationContext."""
    warnings_list = _validation_warnings_ctx.get()
    if warnings_list is None:
        return
    message_str = str(message)
    if any(isinstance(existing, dict) and existing.get("msg") == message_str for existing in warnings_list):
        return
    warnings_list.append(
        {
            "loc": (),
            "msg": message_str,
            "type": "value_error",
            "ctx": {},
        }
    )
