from __future__ import annotations

from typing import Any

from ..core.resolver import CallableResolver

def _unit_list():
    import unyt

    unit_symbols = set()

    for key, value in unyt.unit_symbols.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            unit_symbols.add(str(value))

    return list(unit_symbols)

def _import_flow360(name: str) -> Any:
    import flow360 as fl

    """Import and return a flow360 callable"""
    if name == "fl":
        return fl

    if name == "u":
        from flow360 import u

        return u

WHITELISTED_CALLABLES = {
    "flow360": {"prefix": "u.", "callables": _unit_list()},
}

# Define allowed modules
ALLOWED_MODULES = {
    "flow360", # Flow360 module
    "fl" # For the flow360 alias
}

ALLOWED_CALLABLES = {
    "fl": None,
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
    },
}

IMPORT_FUNCTIONS = {
    ("fl", "u"): _import_flow360,
}

resolver = CallableResolver(ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS)