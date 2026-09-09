"""Shared utilities for the Flow360 Sphinx autodoc extension.

Provides:
- Class introspection helpers (MRO walking, member classification)
- Data collectors for attributes, properties, and methods
- Default value formatting (unyt_quantity, pydantic models)
- Signature simplification for the autodoc-process-signature hook
- Docstring processing via Napoleon NumpyDocstring
"""

from __future__ import annotations

import inspect
import re
import types
from typing import Any

import pydantic as pd
from sphinx.ext.napoleon import Config as NapoleonConfig
from sphinx.ext.napoleon import NumpyDocstring
from sphinx.util.docstrings import prepare_docstring

from flow360_filters import should_skip_member

try:
    import unyt

    _HAS_UNYT = True
except ImportError:
    _HAS_UNYT = False


# ---------------------------------------------------------------------------
# Napoleon configuration
# ---------------------------------------------------------------------------

_NAPOLEON_CFG = NapoleonConfig(
    napoleon_use_param=True,
    napoleon_use_rtype=True,
    napoleon_use_ivar=False,
    napoleon_preprocess_types=False,
)


# ---------------------------------------------------------------------------
# Constant sets
# ---------------------------------------------------------------------------

_BASEMODEL_MEMBERS: frozenset[str] = frozenset(
    name for name in vars(pd.BaseModel) if not name.startswith("__") or name in ("__init__",)
)

FLOW360_BASE_ALLOWED_METHODS: frozenset[str] = frozenset({
    "help",
    "to_file",
    "from_file",
})

PARAM_MODEL_BASE_FIELDS: frozenset[str] = frozenset({
    "version",
    "unit_system",
})

SKIP_PROPERTY_NAMES: frozenset[str] = frozenset({
    "model_extra",
    "model_fields_set",
    "model_computed_fields",
})

SKIPPED_DUNDERS: frozenset[str] = frozenset({
    "__init__",
    "__new__",
    "__str__",
    "__repr__",
    "__hash__",
    "__eq__",
    "__enter__",
    "__exit__",
})

RESOURCE_INTERNAL_METHODS: frozenset[str] = frozenset({
    "create_multipart_upload",
    "upload_part",
    "complete_multipart_upload",
    "get_cloud_path_prefix",
    "is_cloud_resource",
    "get_download_file_list",
})

STOP_CLASSES: frozenset[str] = frozenset({
    "object",
    "ABC",
    "AbstractContextManager",
    "RestApi",
    "Flow360Resource",
    "AssetBase",
    "BaseModel",
    "Flow360BaseModel",
})


# ---------------------------------------------------------------------------
# Class introspection helpers
# ---------------------------------------------------------------------------


def is_flow360_model(cls) -> bool:
    """Return True if *cls* is a subclass of Flow360BaseModel."""
    try:
        from flow360.component.simulation.framework.base_model import Flow360BaseModel

        return issubclass(cls, Flow360BaseModel)
    except (ImportError, TypeError):
        return False


def should_stop_at(klass) -> bool:
    """Return True if MRO walking should stop at *klass* (non-pydantic classes)."""
    return klass.__name__ in STOP_CLASSES


def is_basemodel_member(name: str) -> bool:
    """Return True if *name* is defined directly on pydantic BaseModel."""
    return name in _BASEMODEL_MEMBERS


def is_flow360base_method(name: str) -> bool:
    """Return True if *name* is defined on Flow360BaseModel (not a subclass)."""
    try:
        from flow360.component.simulation.framework.base_model import Flow360BaseModel
    except ImportError:
        return False
    return name in vars(Flow360BaseModel)


def should_skip_flow360base_method(name: str) -> bool:
    """Skip methods from Flow360BaseModel unless they are in the allowed set."""
    if is_flow360base_method(name):
        return name not in FLOW360_BASE_ALLOWED_METHODS
    return False


def is_param_model_base_field(name: str, cls) -> bool:
    """Return True if *name* is a field inherited from _ParamModelBase."""
    if name not in PARAM_MODEL_BASE_FIELDS:
        return False
    for klass in cls.__mro__:
        if klass.__name__ == "_ParamModelBase" and name in getattr(klass, "model_fields", {}):
            return True
    return False


def is_constructor_method(name: str, obj, cls) -> bool:
    """Decide whether a classmethod/staticmethod is a constructor.

    A method is considered a constructor when its return annotation resolves
    to the documented class itself (or one of its bases in the MRO), or —
    when no annotation exists — the method name starts with ``from_``.
    """
    func = obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else obj
    try:
        ret = inspect.signature(func).return_annotation
    except (ValueError, TypeError):
        ret = inspect.Parameter.empty

    if ret is inspect.Parameter.empty:
        return name.startswith("from_")

    if isinstance(ret, str):
        ret_name = ret
    elif hasattr(ret, "__name__"):
        ret_name = ret.__name__
    else:
        return False

    if ret_name == cls.__name__ or ret_name == "Self":
        return True

    for base in cls.__mro__:
        if base.__name__ == ret_name:
            return True

    return False


def is_callable_member(obj) -> bool:
    """Return True if *obj* is a callable member (function, classmethod, staticmethod)."""
    return isinstance(obj, (classmethod, staticmethod, types.FunctionType))


def collect_validator_names(cls) -> frozenset[str]:
    """Gather all validator method names from pydantic's internal decorator registry."""
    names: set[str] = set()
    for klass in cls.__mro__:
        decs = getattr(klass, "__pydantic_decorators__", None)
        if decs is None:
            continue
        for registry_attr in (
            "field_validators",
            "model_validators",
            "validators",
            "root_validators",
        ):
            registry = getattr(decs, registry_attr, None)
            if registry:
                for dec in registry.values():
                    var_name = getattr(dec, "cls_var_name", None) or getattr(dec, "name", None)
                    if var_name:
                        names.add(var_name)
    return frozenset(names)


# ---------------------------------------------------------------------------
# Data collectors — return data, do not emit RST
# ---------------------------------------------------------------------------


def collect_attributes(cls) -> list[tuple[str, Any]]:
    """Return a list of ``(name, FieldInfo)`` for documentable pydantic model fields."""
    model_fields = getattr(cls, "model_fields", None)
    if not model_fields:
        return []

    entries = []
    for name, field_info in model_fields.items():
        if should_skip_member(name, field_info.annotation, field_info):
            continue
        if is_param_model_base_field(name, cls):
            continue
        entries.append((name, field_info))
    return entries


def collect_properties(cls) -> list[tuple[str, property]]:
    """Return a list of ``(name, descriptor)`` for documentable properties.

    Includes both ``property`` and ``functools.cached_property`` descriptors.
    """
    from functools import cached_property

    model = is_flow360_model(cls)
    props: list[tuple[str, property]] = []
    seen: set[str] = set()

    for klass in cls.__mro__:
        if not model and should_stop_at(klass):
            break
        for name, obj in vars(klass).items():
            if name in seen:
                continue
            seen.add(name)
            if not isinstance(obj, (property, cached_property)):
                continue
            if should_skip_member(name):
                continue
            if name in SKIP_PROPERTY_NAMES:
                continue
            if is_basemodel_member(name):
                continue
            props.append((name, obj))
    return props


def collect_methods(cls) -> tuple[list[str], list[str]]:
    """Walk MRO and split callable members into constructors vs methods.

    Returns ``(constructor_names, method_names)``.  For Flow360BaseModel
    subclasses, allowed base methods (help, to_file, from_file) are
    appended at the end of their respective list.
    """
    model = is_flow360_model(cls)
    validator_names = collect_validator_names(cls) if model else frozenset()

    constructors: list[str] = []
    methods: list[str] = []
    base_constructors: list[str] = []
    base_methods: list[str] = []
    seen: set[str] = set()

    for klass in cls.__mro__:
        if not model and should_stop_at(klass):
            break
        for name, obj in vars(klass).items():
            if name in seen:
                continue
            seen.add(name)
            if should_skip_member(name):
                continue
            if name in SKIPPED_DUNDERS:
                continue
            if name in validator_names:
                continue
            if is_basemodel_member(name):
                continue
            if name in RESOURCE_INTERNAL_METHODS:
                continue
            if model and should_skip_flow360base_method(name):
                continue
            if not is_callable_member(obj):
                continue

            is_from_base = model and name in FLOW360_BASE_ALLOWED_METHODS
            if is_constructor_method(name, obj, cls):
                (base_constructors if is_from_base else constructors).append(name)
            else:
                (base_methods if is_from_base else methods).append(name)

    constructors.extend(base_constructors)
    methods.extend(base_methods)
    return constructors, methods


# ---------------------------------------------------------------------------
# Default value formatting
# ---------------------------------------------------------------------------

_UNYT_REPR_RE = re.compile(r"unyt_quantity\(([^,]+),\s*'([^']+)'\)")
_UNIT_TOKEN_RE = re.compile(r"([A-Za-z]+)")


def _prefix_units(unit_str: str) -> str:
    """Prefix every base unit token with ``fl.u.``.

    ``'kg/m**3'`` → ``'fl.u.kg/fl.u.m**3'``
    """
    return _UNIT_TOKEN_RE.sub(r"fl.u.\1", unit_str)
_MODEL_REPR_RE = re.compile(r"(\b[A-Z][A-Za-z0-9_]*)\([^()]*(?:\([^()]*\)[^()]*)*\)")


def format_default_value(value: Any) -> str:
    """Produce a concise, human-readable string for a default value.

    - ``unyt_quantity(288.15, 'K')``  →  ``288.15 * fl.u.K``
    - ``ThermalState(type_name=…)``   →  ``ThermalState()``
    - Plain values use ``repr()``
    """
    if value is None:
        return "None"

    if _HAS_UNYT and isinstance(value, unyt.unyt_quantity):
        return f"{value.value} * {_prefix_units(str(value.units))}"

    if isinstance(value, pd.BaseModel):
        return f"{type(value).__name__}()"

    raw = repr(value)
    raw = _UNYT_REPR_RE.sub(lambda m: f"{m.group(1)} * {_prefix_units(m.group(2))}", raw)
    raw = _simplify_model_in_repr(raw)
    return raw


def _simplify_model_in_repr(text: str) -> str:
    """Collapse ``ModelName(args…)`` to ``ModelName()`` when args contain ``=``."""

    def _replacer(m: re.Match) -> str:
        full = m.group(0)
        name = m.group(1)
        inner = full[len(name) + 1 : -1]
        if "=" in inner:
            return f"{name}()"
        return full

    prev = None
    result = text
    while result != prev:
        prev = result
        result = _MODEL_REPR_RE.sub(_replacer, result)
    return result


# ---------------------------------------------------------------------------
# Signature simplification (for autodoc-process-signature hook)
# ---------------------------------------------------------------------------

_UNYT_SIG_RE = re.compile(r"unyt_quantity\(([^,]+),\s*'([^']+)'\)")
_MODEL_SIG_RE = re.compile(r"(\b[A-Z][A-Za-z0-9_]*)\([^()]*(?:\([^()]*\)[^()]*)*\)")


def simplify_signature(sig: str) -> str:
    """Clean up default values in a method signature string."""
    sig = _UNYT_SIG_RE.sub(r"\1 * fl.u.\2", sig)
    prev = None
    while sig != prev:
        prev = sig
        sig = _MODEL_SIG_RE.sub(
            lambda m: f"{m.group(1)}()"
            if "=" in m.group(0)[len(m.group(1)) + 1 : -1]
            else m.group(0),
            sig,
        )
    return sig


def process_autodoc_signature(app, what, name, obj, options, signature, return_annotation):
    """``autodoc-process-signature`` hook: simplify unyt_quantity and model defaults."""
    if signature:
        signature = simplify_signature(signature)
    return signature, return_annotation


# ---------------------------------------------------------------------------
# Docstring processing via Napoleon
# ---------------------------------------------------------------------------


def process_docstring_napoleon(docstring: str, indent: str = "   ") -> list[str]:
    """Convert a numpydoc-style docstring to RST field lists using Napoleon.

    Uses Sphinx's ``prepare_docstring`` to correctly handle mixed first-line
    indentation, then runs Napoleon's NumpyDocstring converter.

    Returns a list of indented RST lines ready to be emitted inside a
    Sphinx directive body.
    """
    clean = "\n".join(prepare_docstring(docstring))
    parsed = str(NumpyDocstring(clean, config=_NAPOLEON_CFG))
    result: list[str] = []
    for line in parsed.splitlines():
        result.append(f"{indent}{line}" if line.strip() else "")
    return result
