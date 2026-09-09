"""
Utilities for resolving Flow360's custom unit types into human-readable
strings for Sphinx autodoc documentation.

Flow360 uses a custom unit system built on ``unyt``.  Public type aliases
(``DensityType``, ``VelocityType``, …) are ``Annotated`` wrappers around
internal ``_DimensionedType`` subclasses.  Constrained (``.Positive``,
``.NonNegative``, …) and vector/matrix variants are created dynamically at
runtime via classproperties, so standard ``typing`` introspection alone
cannot recover a readable name.  This module bridges that gap.
"""

from __future__ import annotations

import types as _types
from typing import Annotated, Literal, Union, get_args, get_origin

# ---------------------------------------------------------------------------
# Python 3.10+ union syntax (``X | Y``) produces ``types.UnionType``
# ---------------------------------------------------------------------------
_UNION_ORIGINS: set = {Union}
if hasattr(_types, "UnionType"):
    _UNION_ORIGINS.add(_types.UnionType)

# ---------------------------------------------------------------------------
# Lazy-initialised module-level caches
# ---------------------------------------------------------------------------
_DIM_NAME_MAP: dict[str, str] | None = None
_DIM_NAMESPACE_MAP: dict[str, str] | None = None
_DIMENSIONED_TYPE_BASE = None
_VALUE_OR_EXPRESSION_CLS = None
_SENTINEL = type("_Sentinel", (), {})


def _import_dimensioned_type():
    global _DIMENSIONED_TYPE_BASE
    if _DIMENSIONED_TYPE_BASE is None:
        from flow360.component.simulation.unit_system import (
            _DimensionedType,
        )

        _DIMENSIONED_TYPE_BASE = _DimensionedType
    return _DIMENSIONED_TYPE_BASE


def _import_value_or_expression():
    global _VALUE_OR_EXPRESSION_CLS
    if _VALUE_OR_EXPRESSION_CLS is None:
        try:
            from flow360.component.simulation.user_code.core.types import (
                ValueOrExpression,
            )

            _VALUE_OR_EXPRESSION_CLS = ValueOrExpression
        except ImportError:
            _VALUE_OR_EXPRESSION_CLS = _SENTINEL
    return _VALUE_OR_EXPRESSION_CLS


# ---------------------------------------------------------------------------
# Safe subclass test
# ---------------------------------------------------------------------------


def _safe_issubclass(obj, base) -> bool:
    """``issubclass`` that tolerates non-class operands.

    On Python 3.10 ``isinstance(list[int], type)`` is ``True`` even though a
    parametrized generic alias is not a real class, so a bare
    ``isinstance(x, type) and issubclass(x, base)`` guard still reaches
    ``issubclass`` and raises ``TypeError: issubclass() arg 1 must be a
    class``.  (On 3.11+ the ``isinstance`` guard already returns ``False``.)
    Swallow that case so callers can treat such inputs as "not a subclass".
    """
    try:
        return issubclass(obj, base)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# dim_name -> public alias mapping
# ---------------------------------------------------------------------------


def build_dim_name_map() -> dict[str, str]:
    """
    Inspect ``flow360.component.simulation.unit_system`` for module-level
    ``Annotated`` type aliases whose first argument is a ``_DimensionedType``
    subclass.

    Returns a dict like ``{"density": "DensityType", ...}``.
    """
    import flow360.component.simulation.unit_system as mod

    _DimensionedType = _import_dimensioned_type()
    mapping: dict[str, str] = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if get_origin(obj) is not Annotated:
            continue
        args = get_args(obj)
        if not args:
            continue
        first = args[0]
        if (
            _safe_issubclass(first, _DimensionedType)
            and getattr(first, "dim_name", None) is not None
        ):
            mapping[first.dim_name] = name
    return mapping


def _get_dim_name_map() -> dict[str, str]:
    global _DIM_NAME_MAP
    if _DIM_NAME_MAP is None:
        try:
            _DIM_NAME_MAP = build_dim_name_map()
        except Exception:
            _DIM_NAME_MAP = {}
    return _DIM_NAME_MAP


# ---------------------------------------------------------------------------
# dim_name -> namespace class name (new flow360_schema dimension system)
# ---------------------------------------------------------------------------


def build_dimension_namespace_map() -> dict[str, str]:
    """
    Inspect ``flow360_schema.framework.physical_dimensions`` for
    ``PhysicalDimensionBase`` subclasses and map each dimension's
    ``physical_dimension_meta.name`` to its namespace class name.

    Returns a dict like ``{"length": "Length", "temperature":
    "AbsoluteTemperature", "delta_temperature": "DeltaTemperature", ...}``.
    The class name is the public, source-facing spelling users write
    (``Length.Vector3``), so it cannot be derived from ``name`` alone —
    e.g. ``"temperature"`` maps to ``AbsoluteTemperature``.
    """
    import flow360_schema.framework.physical_dimensions as mod
    from flow360_schema.framework.physical_dimensions.dimension_base import (
        PhysicalDimensionBase,
    )

    mapping: dict[str, str] = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if _safe_issubclass(obj, PhysicalDimensionBase) and obj is not PhysicalDimensionBase:
            meta = getattr(obj, "physical_dimension_meta", None)
            dim_name = getattr(meta, "name", None)
            if dim_name:
                mapping[dim_name] = obj.__name__
    return mapping


def _get_dim_namespace_map() -> dict[str, str]:
    global _DIM_NAMESPACE_MAP
    if _DIM_NAMESPACE_MAP is None:
        try:
            _DIM_NAMESPACE_MAP = build_dimension_namespace_map()
        except Exception:
            _DIM_NAMESPACE_MAP = {}
    return _DIM_NAMESPACE_MAP


# ---------------------------------------------------------------------------
# Closure / JSON-schema introspection helpers
# ---------------------------------------------------------------------------


def _extract_closure_var(func, var_name):
    """Return the value of the free variable *var_name* captured by *func*."""
    if func is None or not hasattr(func, "__code__") or func.__closure__ is None:
        return None
    freevars = func.__code__.co_freevars
    if var_name not in freevars:
        return None
    idx = freevars.index(var_name)
    try:
        return func.__closure__[idx].cell_contents
    except (ValueError, IndexError):
        return None


class _MockSchemaHandler:
    """Minimal stand-in for ``pydantic.GetJsonSchemaHandler``."""

    @staticmethod
    def resolve_ref_schema(schema):
        return schema


def _dim_name_via_json_schema(cls_obj) -> str | None:
    """Call ``__get_pydantic_json_schema__`` with a mock handler and read
    ``properties.units.dimension`` from the result."""
    try:
        schema = cls_obj.__get_pydantic_json_schema__(None, _MockSchemaHandler())
        return schema.get("properties", {}).get("units", {}).get("dimension")
    except Exception:
        return None


def _dim_name_via_closure(cls_obj) -> str | None:
    """Walk the closure chain of a ``_Constrained`` JSON-schema lambda to
    locate ``dim_type`` and read its ``dim_name``.

    The lambda's closure contains the inner ``__get_pydantic_json_schema__``
    function whose own closure captures ``dim_type``."""
    func = getattr(cls_obj, "__get_pydantic_json_schema__", None)
    if func is None or not hasattr(func, "__closure__") or func.__closure__ is None:
        return None
    for cell in func.__closure__:
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if not callable(contents) or not hasattr(contents, "__code__"):
            continue
        dim_type = _extract_closure_var(contents, "dim_type")
        if dim_type is not None:
            return getattr(dim_type, "dim_name", None)
    return None


# ---------------------------------------------------------------------------
# Suffix resolvers for dynamically created wrapper types
# ---------------------------------------------------------------------------


def _constraint_suffix(cls_obj) -> str:
    """Map constraint metadata to a human-readable suffix."""
    try:
        interval = cls_obj.con_type.model_fields["value"].metadata[0]
        vals = {
            k: v
            for k, v in [
                ("gt", interval.gt),
                ("ge", interval.ge),
                ("lt", interval.lt),
                ("le", interval.le),
            ]
            if v is not None
        }
    except (AttributeError, KeyError, IndexError):
        return ".Constrained"

    _KNOWN: list[tuple[dict, str]] = [
        ({"gt": 0}, ".Positive"),
        ({"ge": 0}, ".NonNegative"),
        ({"lt": 0}, ".Negative"),
        ({"le": 0}, ".NonPositive"),
    ]
    for pattern, suffix in _KNOWN:
        if vals == pattern:
            return suffix

    return f".Constrained({', '.join(f'{k}={v}' for k, v in vals.items())})"


def _vector_suffix(cls_obj) -> str:
    """Determine the vector/array classproperty name that produced *cls_obj*."""
    length = _extract_closure_var(
        getattr(cls_obj, "__get_pydantic_json_schema__", None), "length"
    )
    dec = getattr(cls_obj, "allow_decreasing", True)
    neg = getattr(cls_obj, "allow_negative_value", True)
    zc = getattr(cls_obj, "allow_zero_component", True)
    zn = getattr(cls_obj, "allow_zero_norm", True)

    if length is None:
        if not neg and not zc:
            return ".PositiveArray"
        if not neg:
            return ".NonNegativeArray"
        return ".Array"
    if length == 2:
        if not dec:
            return ".PositiveRange" if not neg else ".Range"
        return ".Pair"
    if length == 3:
        if not zn and not zc:
            return ".Moment"
        if not zn:
            return ".Direction"
        if not zc and not neg:
            return ".PositiveVector"
        return ".Vector"
    return f".Vector(length={length})"


def _matrix_suffix(cls_obj) -> str:
    shape = _extract_closure_var(
        getattr(cls_obj, "__get_pydantic_json_schema__", None), "shape"
    )
    if shape == (None, 3):
        return ".CoordinateGroup"
    if shape == (3, None):
        return ".CoordinateGroupTranspose"
    return ".Matrix"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_composed_dimension_type(annotation) -> str | None:
    """Resolve a *new* ``flow360_schema`` composed dimension type.

    Since the ``flow360_schema.framework.physical_dimensions`` migration, a
    type such as ``Length.Vector3`` is built by ``_compose_type`` as
    ``Annotated[Any, BeforeValidator(validate), PlainSerializer(...),
    WithJsonSchema(...)]``.  The underlying type is plain ``Any``; the
    dimension and shape survive only inside the ``validate`` closure, which
    captures ``physical_dimension_meta`` and ``data_type``.

    Return the source-facing name (e.g. ``"Length.Vector3"``,
    ``"Velocity.PositiveFloat64"``) or ``None`` if *annotation* is not one of
    these composed types.  ``data_type.name`` already equals the public
    classproperty suffix, so no suffix table is needed.
    """
    if get_origin(annotation) is not Annotated:
        return None
    metadata = get_args(annotation)[1:]
    for meta_obj in metadata:
        func = getattr(meta_obj, "func", None)
        if func is None:
            continue
        pdm = _extract_closure_var(func, "physical_dimension_meta")
        data_type = _extract_closure_var(func, "data_type")
        if pdm is None or data_type is None:
            continue
        dim_name = getattr(pdm, "name", None)
        type_name = getattr(data_type, "name", None)
        if not dim_name or not type_name:
            continue
        namespace = _get_dim_namespace_map().get(dim_name)
        if namespace:
            return f"{namespace}.{type_name}"
    return None


def resolve_unit_type(annotation) -> str | None:
    """If *annotation* represents a Flow360 unit type return a human-readable
    name such as ``"Length.Vector3"`` (new system) or ``"DensityType.Positive"``
    (legacy system); otherwise return ``None``."""
    # New flow360_schema composed types are checked first so that resolution
    # works even if the legacy ``unit_system`` module is unavailable.
    composed = _resolve_composed_dimension_type(annotation)
    if composed is not None:
        return composed

    try:
        _DimensionedType = _import_dimensioned_type()
    except ImportError:
        return None

    dim_map = _get_dim_name_map()

    inner = annotation
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if not args:
            return None
        inner = args[0]

    if _safe_issubclass(inner, _DimensionedType):
        dim_name = getattr(inner, "dim_name", None)
        return dim_map.get(dim_name) if dim_name else None

    cls_name = getattr(inner, "__name__", None)

    if cls_name == "_Constrained" and hasattr(inner, "con_type"):
        dim_name = _dim_name_via_json_schema(inner) or _dim_name_via_closure(inner)
        base = dim_map.get(dim_name) if dim_name else None
        return f"{base}{_constraint_suffix(inner)}" if base else None

    # The dynamic class is named "_VectorType" in current source; guard
    # against an older/alternate name ("_VectorValidator") as well.
    if cls_name in ("_VectorType", "_VectorValidator"):
        dim_type = getattr(inner, "type", None)
        dim_name = getattr(dim_type, "dim_name", None) if dim_type else None
        base = dim_map.get(dim_name) if dim_name else None
        return f"{base}{_vector_suffix(inner)}" if base else None

    if cls_name == "_MatrixType":
        dim_type = getattr(inner, "type", None)
        dim_name = getattr(dim_type, "dim_name", None) if dim_type else None
        base = dim_map.get(dim_name) if dim_name else None
        return f"{base}{_matrix_suffix(inner)}" if base else None

    return None


def format_type_annotation(annotation) -> str:
    """Render *annotation* as a concise, human-readable string.

    Flow360 unit types are resolved to their public alias names; standard
    ``typing`` constructs (``Optional``, ``Union``, ``Literal``, generics)
    are formatted recursively.
    """
    if annotation is type(None):
        return "None"
    if isinstance(annotation, str):
        return annotation

    unit = resolve_unit_type(annotation)
    if unit is not None:
        return unit

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"

    if origin in _UNION_ORIGINS:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        if len(non_none) == 1 and has_none:
            return f"{format_type_annotation(non_none[0])}, optional"
        inner = ", ".join(format_type_annotation(a) for a in non_none)
        return f"Union[{inner}], optional" if has_none else f"Union[{inner}]"

    # Annotated[X, ...] that wasn't caught as a unit type
    if origin is Annotated:
        return format_type_annotation(args[0]) if args else str(annotation)

    VoE = _import_value_or_expression()
    if VoE is not _SENTINEL:
        if origin is VoE or (isinstance(origin, type) and issubclass(origin, VoE)):
            if args:
                return f"ValueOrExpression[{format_type_annotation(args[0])}]"
            return "ValueOrExpression"
        if annotation is VoE:
            return "ValueOrExpression"

    if origin is not None and args:
        if isinstance(origin, type):
            name = _qualified_name(origin)
        else:
            name = (
                getattr(origin, "__name__", None)
                or getattr(origin, "_name", None)
                or str(origin)
            )
        return f"{name}[{', '.join(format_type_annotation(a) for a in args)}]"

    if isinstance(annotation, type):
        return _qualified_name(annotation)

    return str(annotation)


def _metadata_has_unit_validator(metadata) -> bool:
    """True if *metadata* carries a ``flow360_schema`` composed-unit
    ``BeforeValidator`` — i.e. the field's *top-level* ``Annotated`` unit type
    was stripped by Pydantic onto ``FieldInfo.metadata``, leaving the
    annotation as bare ``Any``."""
    for meta_obj in metadata:
        func = getattr(meta_obj, "func", None)
        if func is None:
            continue
        if (
            _extract_closure_var(func, "physical_dimension_meta") is not None
            and _extract_closure_var(func, "data_type") is not None
        ):
            return True
    return False


def format_field_annotation(field_info) -> str:
    """Render a Pydantic ``FieldInfo``'s type as a human-readable string.

    Pydantic strips a *top-level* ``Annotated[...]`` field type down to its
    underlying type on ``FieldInfo.annotation`` and moves the annotation
    metadata to ``FieldInfo.metadata``.  For Flow360's new ``flow360_schema``
    composed dimension types the underlying type is plain ``Any``, so the
    dimension and shape survive *only* in the metadata — formatting
    ``field_info.annotation`` alone would render ``Any``.  Recombine the two so
    such fields render as e.g. ``Length.Vector3``.

    Only the genuinely-stripped top-level case is recombined.  Nested types
    (``list[Length.Vector3]``, ``Optional[Length.Vector3]``) keep their
    ``Annotated`` *inside* the generic — the annotation is already complete, so
    re-wrapping it is both unnecessary and harmful: a field like
    ``list[Length.Vector3]`` carrying its own ``MinLen`` metadata would become
    ``Annotated[list[Annotated[Any, ...]], MinLen(...)]``, which Python 3.10's
    typing cannot format (it raised, and the documenter fell back to the raw
    ``str`` repr on Read the Docs).  Detecting the unit ``BeforeValidator`` in
    the metadata distinguishes the stripped case from incidental metadata.
    """
    annotation = field_info.annotation
    metadata = list(getattr(field_info, "metadata", None) or [])
    if metadata and _metadata_has_unit_validator(metadata):
        try:
            annotation = Annotated[tuple([annotation, *metadata])]
        except Exception:
            annotation = field_info.annotation
    return format_type_annotation(annotation)


def _qualified_name(cls: type) -> str:
    """Return a fully qualified name with Sphinx ``~`` prefix for abbreviation.

    ``~module.ClassName`` resolves unambiguously but renders as just
    ``ClassName`` in the output.  Built-in types (``str``, ``int``, …)
    are returned as bare names since they need no disambiguation.
    """
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", cls.__name__)
    if module and module != "builtins" and "<" not in qualname:
        return f"~{module}.{qualname}"
    return cls.__name__
