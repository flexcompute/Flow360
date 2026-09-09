"""Context-aware validator decorators for Flow360 schemas."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import pydantic as pd

from .context import ALL, get_validation_info, get_validation_levels

_MISSING_SIGNATURE = object()


def _unwrap_descriptor(func: Any) -> tuple[Callable[..., Any], bool, bool]:
    """Return the underlying callable and descriptor metadata."""
    is_classmethod = isinstance(func, classmethod)
    is_staticmethod = isinstance(func, staticmethod)
    original_func = func.__func__ if is_classmethod or is_staticmethod else func
    return original_func, is_classmethod, is_staticmethod


def _patch_param_info_signature(func: Callable[..., Any]) -> tuple[bool, Any]:
    """Temporarily remove ``param_info`` from function signature for Pydantic."""
    original_sig = inspect.signature(func)
    pass_param_info = "param_info" in original_sig.parameters
    original_signature_backup = getattr(func, "__signature__", _MISSING_SIGNATURE)

    if pass_param_info:
        params_without = tuple(param for name, param in original_sig.parameters.items() if name != "param_info")
        func.__signature__ = original_sig.replace(parameters=params_without)  # type: ignore[attr-defined]  # Dynamic signature modification is valid at runtime

    return pass_param_info, original_signature_backup


def _restore_signature(func: Callable[..., Any], pass_param_info: bool, signature_backup: Any) -> None:
    """Restore original function signature after wrapper creation."""
    if not pass_param_info:
        return

    if signature_backup is _MISSING_SIGNATURE:
        if hasattr(func, "__signature__"):
            del func.__signature__
    else:
        func.__signature__ = signature_backup  # type: ignore[attr-defined]  # Dynamic signature modification is valid at runtime


def _rewrap_descriptor(func: Callable[..., Any], is_classmethod: bool, is_staticmethod: bool) -> Any:
    """Re-apply classmethod/staticmethod if the input was a descriptor."""
    if is_classmethod:
        return classmethod(func)
    if is_staticmethod:
        return staticmethod(func)
    return func


def _get_skip_return_value(args: tuple[Any, ...]) -> Any:
    """Return the original value/object when contextual validation is skipped."""
    if not args:
        return None
    value_idx = 1 if isinstance(args[0], type) and len(args) >= 2 else 0
    return args[value_idx]


def context_validator(
    context: Literal["SurfaceMesh", "VolumeMesh", "Case"],
) -> Callable[..., Any]:
    """Decorator to conditionally run a validator based on validation context.

    The decorated validator only runs if the current validation level matches
    *context* or is ALL.
    """

    def decorator(func: Any) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            current_levels = get_validation_levels()
            if current_levels is None or any(lvl in (context, ALL) for lvl in current_levels):
                return func(self, *args, **kwargs)
            return self

        return wrapper

    return decorator


def contextual_field_validator(
    *fields: str,
    mode: str = "after",
    required_context: list[str] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Wrapper around ``pydantic.field_validator`` that skips when no validation info is set.

    Parameters
    ----------
    *fields : str
        Field names to validate.
    mode : str
        Validation mode (``"before"``, ``"after"``, ``"wrap"``).
    required_context : list[str] | None
        Attribute names on the validation info object that must be not-None
        for the validator to run.
    """

    def decorator(func: Any) -> Callable[..., Any]:
        original_func, is_classmethod, is_staticmethod = _unwrap_descriptor(func)
        pass_param_info, original_signature_backup = _patch_param_info_signature(original_func)

        @wraps(original_func)
        def wrapper(*args: Any, **kwargs_inner: Any) -> Any:
            param_info = get_validation_info()
            if param_info is None:
                return _get_skip_return_value(args)

            if required_context:
                for attr_name in required_context:
                    if not hasattr(param_info, attr_name):
                        raise ValueError(f"Invalid validation context attribute: {attr_name}")
                    if getattr(param_info, attr_name) is None:
                        return _get_skip_return_value(args)

            call_kwargs = dict(kwargs_inner)
            if pass_param_info:
                call_kwargs["param_info"] = param_info
            return original_func(*args, **call_kwargs)

        _restore_signature(original_func, pass_param_info, original_signature_backup)
        wrapped_func = _rewrap_descriptor(wrapper, is_classmethod, is_staticmethod)

        return pd.field_validator(*fields, mode=mode, **kwargs)(wrapped_func)  # type: ignore[call-overload,no-any-return]  # Dynamic mode string is valid at runtime

    return decorator


def contextual_model_validator(mode: str = "after", **kwargs: Any) -> Callable[..., Any]:
    """Wrapper around ``pydantic.model_validator`` that skips when no validation info is set."""

    def decorator(func: Any) -> Callable[..., Any]:
        original_func, is_classmethod, is_staticmethod = _unwrap_descriptor(func)
        pass_param_info, original_signature_backup = _patch_param_info_signature(original_func)

        @wraps(original_func)
        def wrapper(*args: Any, **kwargs_inner: Any) -> Any:
            param_info = get_validation_info()
            if param_info is None:
                return _get_skip_return_value(args)
            call_kwargs = dict(kwargs_inner)
            if pass_param_info:
                call_kwargs["param_info"] = param_info
            return original_func(*args, **call_kwargs)

        _restore_signature(original_func, pass_param_info, original_signature_backup)
        wrapped_func = _rewrap_descriptor(wrapper, is_classmethod, is_staticmethod)

        return pd.model_validator(mode=mode, **kwargs)(wrapped_func)  # type: ignore[call-overload,no-any-return]  # Dynamic mode string is valid at runtime

    return decorator
