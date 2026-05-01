"""
This module is flow360 for simulation based models.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from types import ModuleType

_PUBLIC_NAMESPACE_MODULE: ModuleType | None = None


def _load_public_namespace_module() -> ModuleType:
    global _PUBLIC_NAMESPACE_MODULE  # pylint: disable=global-statement
    if _PUBLIC_NAMESPACE_MODULE is None:
        _PUBLIC_NAMESPACE_MODULE = import_module("flow360._public_namespace")
    return _PUBLIC_NAMESPACE_MODULE


def _load_exported_names() -> list[str]:
    namespace_source = Path(__file__).with_name("_public_namespace.py").read_text(encoding="utf-8")
    module_ast = ast.parse(namespace_source)
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return ast.literal_eval(node.value)
    return []


__all__ = _load_exported_names()


def __getattr__(name: str):
    if name == "version_check":
        module = import_module("flow360.version_check")
        globals()[name] = module
        return module

    if name in __all__:
        namespace_module = _load_public_namespace_module()
        try:
            value = getattr(namespace_module, name)
        except AttributeError as error:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
