"""Whitelisted functions and classes that can be called from blueprint functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _import_tidy3d(name: str) -> Any:
    """Import and return a tidy3d callable."""
    import tidy3d as td

    if name == "td":
        return td

    if name.startswith("td."):
        # Generate callables dict dynamically from WHITELISTED_CALLABLES
        td_core = WHITELISTED_CALLABLES["tidy3d.core"]
        callables = {
            f"{td_core['prefix']}{name}": getattr(td, name) for name in td_core["callables"]
        }
        return callables[name]
    elif name == "ModeSolver":
        from tidy3d.plugins.mode import ModeSolver

        return ModeSolver
    elif name == "C_0":
        from tidy3d.constants import C_0

        return C_0
    raise ValueError(f"Unknown tidy3d callable: {name}")


def _import_flow360(name: str) -> Any:
    import flow360 as fl

    """Import and return a flow360 callable"""
    if name == "fl":
        return fl

    if name == "u":
        from flow360 import u

        return u


def _unit_list():
    import unyt

    unit_symbols = set()

    for key, value in unyt.unit_symbols.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            unit_symbols.add(str(value))

    return list(unit_symbols)


def _import_utilities(name: str) -> Callable[..., Any]:
    """Import and return a utility callable."""
    from rich import print

    callables = {
        "print": print,
    }
    return callables[name]


# Single source of truth for whitelisted callables
WHITELISTED_CALLABLES = {
    "tidy3d.core": {
        "prefix": "td.",
        "callables": [
            "Medium",
            "GridSpec",
            "Box",
            "Structure",
            "Simulation",
            "BoundarySpec",
            "Periodic",
            "ModeSpec",
            "inf",
        ],
    },
    "tidy3d.plugins": {
        "prefix": "",
        "callables": ["ModeSolver"],
    },
    "tidy3d.constants": {
        "prefix": "",
        "callables": ["C_0"],
    },
    "utilities": {
        "prefix": "",
        "callables": ["print"],
    },
    "flow360": {"prefix": "u.", "callables": _unit_list()},
}

# Define allowed modules
ALLOWED_MODULES = {
    "ModeSolver",  # For the ModeSolver class
    "tidy3d",  # For the tidy3d module
    "td",  # For the tidy3d alias
}

# Generate ALLOWED_CALLABLES from the single source of truth
ALLOWED_CALLABLES = {
    "td": None,
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
    },
}

# Generate import category mapping
_IMPORT_FUNCTIONS = {
    ("fl", "u"): _import_flow360,
    ("td", "td.", "ModeSolver", "C_0"): _import_tidy3d,
    ("print",): _import_utilities,
}


class CallableResolver:
    """Manages resolution and validation of callable objects.

    Provides a unified interface for resolving function names, methods, and
    attributes while enforcing whitelisting rules.
    """

    def __init__(self) -> None:
        self._allowed_callables: dict[str, Callable[..., Any]] = {}
        self._allowed_modules: dict[str, Any] = {}

        # Initialize with safe builtins
        self._safe_builtins = {
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            # Add other safe builtins as needed
        }

    def register_callable(self, name: str, func: Callable[..., Any]) -> None:
        """Register a callable for direct use."""
        self._allowed_callables[name] = func

    def register_module(self, name: str, module: Any) -> None:
        """Register a module for attribute access."""
        self._allowed_modules[name] = module

    def get_callable(
        self,
        qualname: str,
        context: EvaluationContext | None = None,  # noqa: F821
    ) -> Callable[..., Any]:
        """Resolve a callable by its qualified name.

        Args:
            qualname: Fully qualified name (e.g., "np.array" or "len")
            context: Optional evaluation context for local lookups

        Returns:
            The resolved callable object

        Raises:
            ValueError: If the callable is not allowed or cannot be found
        """
        # Try context first if provided
        if context is not None:
            try:
                return context.get(qualname)
            except KeyError:
                pass

        # Check direct allowed callables
        if qualname in self._allowed_callables:
            return self._allowed_callables[qualname]

        # Check safe builtins
        if qualname in self._safe_builtins:
            return self._safe_builtins[qualname]

        # Handle module attributes
        if "." in qualname:
            module_name, *attr_parts = qualname.split(".")
            if module_name in self._allowed_modules:
                obj = self._allowed_modules[module_name]
                for part in attr_parts:
                    obj = getattr(obj, part)
                if qualname in ALLOWED_CALLABLES:
                    return obj
            # Try importing if it's a whitelisted callable
            if qualname in ALLOWED_CALLABLES:
                for names, import_func in _IMPORT_FUNCTIONS.items():
                    if module_name in names:
                        module = import_func(module_name)
                        self.register_module(module_name, module)
                        obj = module
                        for part in attr_parts:
                            obj = getattr(obj, part)
                        return obj

        raise ValueError(f"Callable '{qualname}' is not allowed")


# Create global resolver instance
resolver = CallableResolver()


def get_allowed_callable(
    qualname: str,
    context: EvaluationContext | None = None,  # noqa: F821
) -> Callable[..., Any]:
    """Get an allowed callable by name."""
    # Try getting from resolver first
    try:
        return resolver.get_callable(qualname, context)
    except ValueError as e:
        # Check if it's a whitelisted callable before trying to import
        if (
            qualname in ALLOWED_CALLABLES
            or qualname in ALLOWED_MODULES
            or any(
                qualname.startswith(f"{group['prefix']}{name}")
                for group in WHITELISTED_CALLABLES.values()
                for name in group["callables"]
            )
        ):
            # If found in resolver, try importing on demand
            for names, import_func in _IMPORT_FUNCTIONS.items():
                if qualname in names or any(qualname.startswith(prefix) for prefix in names):
                    callable_obj = import_func(qualname)
                    resolver.register_callable(qualname, callable_obj)
                    return callable_obj
        raise ValueError(f"Callable '{qualname}' is not allowed") from e
