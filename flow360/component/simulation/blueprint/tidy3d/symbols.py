from __future__ import annotations

from typing import Any, Callable

from ..core.resolver import CallableResolver


def _import_utilities(name: str) -> Callable[..., Any]:
    """Import and return a utility callable."""
    from rich import print

    callables = {
        "print": print,
    }
    return callables[name]


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
        "evaluate": True,
    },
    "tidy3d.plugins": {"prefix": "", "callables": ["ModeSolver"], "evaluate": True},
    "tidy3d.constants": {"prefix": "", "callables": ["C_0"], "evaluate": True},
    "utilities": {"prefix": "", "callables": ["print"], "evaluate": True},
}

# Define allowed modules
ALLOWED_MODULES = {
    "ModeSolver",  # For the ModeSolver class
    "tidy3d",  # For the tidy3d module
    "td",  # For the tidy3d alias
}

ALLOWED_CALLABLES = {
    "td": None,
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
    },
}

EVALUATION_BLACKLIST = {
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
        if not group["evaluate"]
    },
}

# Generate import category mapping
IMPORT_FUNCTIONS = {
    ("td", "td.", "ModeSolver", "C_0"): _import_tidy3d,
    ("print",): _import_utilities,
}

resolver = CallableResolver(
    ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS, EVALUATION_BLACKLIST
)
