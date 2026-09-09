"""Entity expansion utilities that depend only on schema-owned types."""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args, get_origin

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)

if TYPE_CHECKING:
    from flow360_schema.framework.base_model import Flow360BaseModel
    from flow360_schema.framework.entity.entity_base import EntityBase
    from flow360_schema.framework.entity.entity_list import EntityList
    from flow360_schema.framework.entity.entity_registry import EntityRegistry


def _get_mapping_or_attribute(obj: Any, name: str) -> Any:
    """Return a value from either a dict-like payload or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _register_mirror_entities_in_registry(registry: EntityRegistry, mirror_status: Any) -> None:
    """Register mirror-related entities (planes + derived mirrored entities) into registry.

    This helper is shared by both dict-based and params-based registry builders to ensure
    consistent selector expansion coverage.
    """
    if not mirror_status:
        return

    # Lazy import to avoid pulling these models unless mirror status is actually present.
    from flow360_schema.models.asset_cache import MirrorStatus
    from flow360_schema.models.entities.geometry_entities import MirrorPlane

    # Dict path: deserialize to MirrorStatus
    if isinstance(mirror_status, dict):
        mirror_status = MirrorStatus.deserialize(mirror_status)

    # Object path: MirrorStatus (or compatible) with is_empty()
    if hasattr(mirror_status, "is_empty") and mirror_status.is_empty():
        return

    for plane in getattr(mirror_status, "mirror_planes", []) or []:
        if isinstance(plane, MirrorPlane):
            registry.register(plane)
    for mirrored_group in getattr(mirror_status, "mirrored_geometry_body_groups", []) or []:
        registry.register(mirrored_group)
    for mirrored_surface in getattr(mirror_status, "mirrored_surfaces", []) or []:
        registry.register(mirrored_surface)


def _register_imported_surfaces_in_registry(registry: EntityRegistry, imported_surfaces: Any) -> None:
    """Register asset-cache imported surfaces so they are resolvable like every other assignable entity."""
    if not imported_surfaces:
        return

    from flow360_schema.models.entities.surface_entities import ImportedSurface

    for item in imported_surfaces:
        if isinstance(item, dict):
            item = ImportedSurface.deserialize(item)
        registry.register(item)


def get_entity_info_and_registry_from_asset_cache(asset_cache: Any) -> tuple[Any, Any]:
    """
    Create EntityInfo and EntityRegistry from an asset cache object or dict.

    The EntityInfo owns the entities, and EntityRegistry holds references to them.
    Callers must keep entity_info alive as long as registry is used.
    """
    from flow360_schema.framework.entity.entity_registry import EntityRegistry
    from flow360_schema.models.entity_info import parse_entity_info_model

    if asset_cache is None:
        raise ValueError("[Internal] asset_cache is required to build entity registry.")

    entity_info = _get_mapping_or_attribute(asset_cache, "project_entity_info")
    if entity_info is None:
        raise ValueError("[Internal] project_entity_info not found in asset cache.")

    if isinstance(entity_info, dict):
        entity_info = parse_entity_info_model(entity_info)

    registry = EntityRegistry.from_entity_info(entity_info)

    mirror_status = _get_mapping_or_attribute(asset_cache, "mirror_status")
    _register_mirror_entities_in_registry(registry, mirror_status)

    imported_surfaces = _get_mapping_or_attribute(asset_cache, "imported_surfaces")
    _register_imported_surfaces_in_registry(registry, imported_surfaces)

    return entity_info, registry


def get_entity_info_and_registry_from_dict(params_as_dict: dict[str, Any]) -> tuple[Any, Any]:
    """
    Create EntityInfo and EntityRegistry from simulation params dictionary.

    The EntityInfo owns the entities, and EntityRegistry holds references to them.
    Callers must keep entity_info alive as long as registry is used.

    Parameters
    ----------
    params_as_dict : dict
        Simulation parameters as dictionary containing private_attribute_asset_cache.

    Returns
    -------
    tuple[EntityInfo, EntityRegistry]
        (entity_info, registry) where entity_info owns entities and registry references them.
    """
    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    if asset_cache is None:
        raise ValueError("[Internal] private_attribute_asset_cache not found in params_as_dict.")
    return get_entity_info_and_registry_from_asset_cache(asset_cache)


def get_registry_from_asset_cache(asset_cache: Any) -> Any:
    """Create an EntityRegistry from an asset cache object or dict."""
    return get_entity_info_and_registry_from_asset_cache(asset_cache)[1]


def expand_entity_list_with_registry(
    entity_list: EntityList,
    registry: Any | None = None,
    *,
    return_names: bool = False,
) -> list[EntityBase] | list[str]:
    """
    Expand selectors for a deserialized EntityList within an EntityRegistry context.

    When no selectors are present, `registry` may be omitted and explicit stored entities
    will still be materialized and returned as-is. Explicit entities are expected to
    have already been filtered by EntityList during validation.
    """
    stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
    selectors = list(getattr(entity_list, "selectors", []) or [])

    if selectors:
        if registry is None:
            raise Flow360ValueError("An EntityRegistry is required to expand selectors in the given EntityList.")

        from flow360_schema.framework.entity.entity_selector import (
            resolve_entity_list_selectors,
        )

        try:
            stored_entities = resolve_entity_list_selectors(
                registry,
                entity_list,
                selector_cache={},
                merge_mode="merge",
            )
        except ValueError as exc:
            raise Flow360ValueError(
                "Failed to find any valid entities in the input. "
                "Has the simulationParams been manually edited since loading from the cloud "
                "or have you changed the cloud resource for which the SimulationParams is being used?"
            ) from exc

    if not stored_entities:
        return []

    if not all(hasattr(entity, "name") for entity in stored_entities):
        wrapper = {"stored_entities": stored_entities}
        materialize_entities_and_selectors_in_place(wrapper)
        stored_entities = wrapper.get("stored_entities", [])

    if return_names:
        return [entity.name for entity in stored_entities]
    return stored_entities


# Which fields can the expansion walk skip? An EntityList is a pydantic model, so it
# enters the params tree only through model fields and the standard containers inside
# them (``extra="forbid"`` and validate-on-assignment keep every value within its
# field's annotation). A field whose annotation cannot reach an EntityList therefore
# cannot hold one — on catalog-heavy models (an entity_info holding hundreds of
# thousands of surfaces) skipping those fields is the difference between walking every
# entity and not entering the field at all.
#
# The verdict errs toward True: a wrong True only costs walk time, a wrong False would
# silently skip a selector expansion. ``Any``, bare containers, and unresolved
# annotations all count as True. Foreign non-model classes (unyt quantities, ndarrays,
# datetimes) count as False: they cannot be an EntityList and pydantic annotations are
# the only doorway for one.
#
# Weak keys for the same reason as the preprocess caches in base_model_utils: each
# EntityList subscription mints a fresh class, and models parametrized on them must
# not be pinned forever.
_entity_list_fields_cache: weakref.WeakKeyDictionary[type, frozenset[str]] = weakref.WeakKeyDictionary()


def _annotation_may_reach_entity_list(annotation: Any, in_progress: set[type]) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Annotated:
        return _annotation_may_reach_entity_list(get_args(annotation)[0], in_progress)
    if origin is Literal:
        return False
    if origin is not None:
        args = tuple(arg for arg in get_args(annotation) if arg is not Ellipsis)
        if not args:
            return True
        return any(_annotation_may_reach_entity_list(arg, in_progress) for arg in args)
    return _class_may_reach_entity_list(annotation, in_progress)


def _class_may_reach_entity_list(cls: Any, in_progress: set[type]) -> bool:
    if not isinstance(cls, type):
        return True
    from flow360_schema.framework.base_model import Flow360BaseModel
    from flow360_schema.framework.entity.entity_list import EntityList

    if issubclass(cls, EntityList):
        return True
    if issubclass(cls, Flow360BaseModel):
        if cls in in_progress:
            # A cycle adds no EntityList leaves beyond those its members already have.
            return False
        in_progress.add(cls)
        try:
            return any(
                _annotation_may_reach_entity_list(field.annotation, in_progress) for field in cls.model_fields.values()
            )
        finally:
            in_progress.discard(cls)
    if issubclass(cls, (list, tuple, set, frozenset, dict)):
        parameterized = tuple(base for base in getattr(cls, "__orig_bases__", ()) if get_args(base))
        if not parameterized:
            return True
        return any(_annotation_may_reach_entity_list(base, in_progress) for base in parameterized)
    return False


def entity_list_reaching_field_names(model_cls: type[Flow360BaseModel]) -> frozenset[str]:
    """Fields of ``model_cls`` whose annotation subtree may hold an EntityList."""
    cached = _entity_list_fields_cache.get(model_cls)
    if cached is None:
        cached = frozenset(
            name
            for name, field in model_cls.model_fields.items()
            if _annotation_may_reach_entity_list(field.annotation, set())
        )
        with contextlib.suppress(TypeError):
            _entity_list_fields_cache[model_cls] = cached
    return cached


def expand_all_entity_lists_with_registry_in_place(
    root_obj: Any,
    *,
    registry: Any,
    merge_mode: Literal["merge", "replace"] = "merge",
    expansion_map: dict[str, list[str]] | None = None,
    expansion_lookup: Callable[[EntityList], list[EntityBase] | None] | None = None,
) -> None:
    """Resolve selectors for all EntityList objects under `root_obj` in-place.

    Models descend only through fields whose annotation may reach an EntityList
    (see ``entity_list_reaching_field_names``), so the entity catalog itself is
    never walked entity-by-entity.

    ``expansion_lookup`` lets a caller reuse expansions already resolved elsewhere
    (validation's per-instance cache): a non-None lookup result is written back
    verbatim instead of re-resolving. It is only consulted for the default
    ``merge_mode="merge"`` / ``expansion_map=None`` combination — the semantics the
    validation-time expansion used — so any other combination resolves from scratch.
    """
    from flow360_schema.framework.base_model import Flow360BaseModel
    from flow360_schema.framework.entity.entity_list import EntityList
    from flow360_schema.framework.entity.entity_selector import (
        resolve_entity_list_selectors,
    )

    if merge_mode != "merge" or expansion_map is not None:
        expansion_lookup = None

    selector_cache: dict[str, Any] = {}
    visited: set[int] = set()

    def _walk(obj: Any) -> None:
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, EntityList):
            resolved_entities = expansion_lookup(obj) if expansion_lookup is not None else None
            if resolved_entities is None:
                resolved_entities = resolve_entity_list_selectors(
                    registry,
                    obj,
                    selector_cache=selector_cache,
                    merge_mode=merge_mode,
                    expansion_map=expansion_map,
                )
            obj.stored_entities = resolved_entities
            return
        if isinstance(obj, Flow360BaseModel):
            for name in entity_list_reaching_field_names(type(obj)):
                _walk(obj.__dict__.get(name))
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value)

    _walk(root_obj)


__all__ = [
    "_register_mirror_entities_in_registry",
    "expand_all_entity_lists_with_registry_in_place",
    "expand_entity_list_with_registry",
    "get_entity_info_and_registry_from_asset_cache",
    "get_entity_info_and_registry_from_dict",
    "get_registry_from_asset_cache",
]
