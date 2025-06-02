"""Whitelisted functions and classes that can be called from blueprint functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class CallableResolver:
    """Manages resolution and validation of callable objects.

    Provides a unified interface for resolving function names, methods, and
    attributes while enforcing whitelisting rules.
    """

    def __init__(self, callables, modules, imports, blacklist) -> None:
        self._import_builtins = imports
        self._callable_builtins = callables
        self._module_builtins = modules
        self._evaluation_blacklist = blacklist

        self._allowed_callables: dict[str, Callable[..., Any]] = {}
        self._allowed_modules: dict[str, Any] = {}

    def register_callable(self, name: str, func: Callable[..., Any]) -> None:
        """Register a callable for direct use."""
        self._allowed_callables[name] = func

    def register_module(self, name: str, module: Any) -> None:
        """Register a module for attribute access."""
        self._allowed_modules[name] = module

    def can_evaluate(self, qualname: str) -> bool:
        """Check if the name is not blacklisted for evaluation by the resolver"""
        return qualname not in self._evaluation_blacklist

    def get_callable(self, qualname: str) -> Callable[..., Any]:
        """Resolve a callable by its qualified name.

        Args:
            qualname: Fully qualified name (e.g., "np.array" or "len")
            context: Optional evaluation context for local lookups

        Returns:
            The resolved callable object

        Raises:
            ValueError: If the callable is not allowed or cannot be found
        """
        # Check direct allowed callables
        if qualname in self._allowed_callables:
            return self._allowed_callables[qualname]

        # Handle module attributes
        if "." in qualname:
            module_name, *attr_parts = qualname.split(".")
            if module_name in self._allowed_modules:
                obj = self._allowed_modules[module_name]
                for part in attr_parts:
                    obj = getattr(obj, part)
                if qualname in self._callable_builtins:
                    return obj
            # Try importing if it's a whitelisted callable
            if qualname in self._callable_builtins:
                for names, import_func in self._import_builtins.items():
                    if module_name in names:
                        module = import_func(module_name)
                        self.register_module(module_name, module)
                        obj = module
                        for part in attr_parts:
                            obj = getattr(obj, part)
                        return obj

        raise ValueError(f"Callable '{qualname}' is not allowed")

    def get_allowed_callable(self, qualname: str) -> Callable[..., Any]:
        """Get an allowed callable by name."""
        try:
            return self.get_callable(qualname)
        except ValueError as e:
            # Check if it's a whitelisted callable before trying to import
            if (
                qualname in self._callable_builtins
                or qualname in self._module_builtins
                or any(
                    qualname.startswith(f"{group['prefix']}{name}")
                    for group in self._callable_builtins.values()
                    if group is not None
                    for name in group["callables"]
                )
            ):
                # If found in resolver, try importing on demand
                for names, import_func in self._import_builtins.items():
                    if qualname in names or any(qualname.startswith(prefix) for prefix in names):
                        callable_obj = import_func(qualname)
                        self.register_callable(qualname, callable_obj)
                        return callable_obj
            raise ValueError(f"Callable '{qualname}' is not allowed") from e
