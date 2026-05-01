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


def _is_all_target(target: ast.expr) -> bool:
    return isinstance(target, ast.Name) and target.id == "__all__"


def _literal_exported_names(value: ast.expr | None) -> list[str]:
    if value is None:
        raise RuntimeError("flow360._public_namespace must assign __all__")
    names = ast.literal_eval(value)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RuntimeError("flow360._public_namespace __all__ must be a list of strings")
    return names


def _extract_exported_names(module_ast: ast.Module) -> list[str]:
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_all_target(target):
                    return _literal_exported_names(node.value)
        if isinstance(node, ast.AnnAssign) and _is_all_target(node.target):
            return _literal_exported_names(node.value)
    raise RuntimeError("flow360._public_namespace must define __all__")


def _load_exported_names() -> list[str]:
    namespace_source = Path(__file__).with_name("_public_namespace.py").read_text(encoding="utf-8")
    module_ast = ast.parse(namespace_source)
    return _extract_exported_names(module_ast)


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
