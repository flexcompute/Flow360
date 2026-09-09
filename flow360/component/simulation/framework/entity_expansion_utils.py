"""Entity list expansion helpers shared across results and user utilities."""

# pylint: disable=unused-import
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_expansion_config import (
    DEFAULT_TARGET_CLASS_EXPANSION_MAP,
)
from flow360_schema.framework.entity.entity_expansion_utils import (  # noqa: F401
    _register_mirror_entities_in_registry,
    expand_all_entity_lists_with_registry_in_place,
    expand_entity_list_with_registry,
    get_entity_info_and_registry_from_asset_cache,
    get_entity_info_and_registry_from_dict,
    get_registry_from_asset_cache,
)
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.entity_registry import EntityRegistryView
from flow360_schema.framework.entity.entity_selector import (
    EntitySelector,
    expand_entity_list_selectors,
)
from flow360_schema.framework.entity.entity_utils import deduplicate_entities

from flow360.exceptions import Flow360ValueError

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_registry import EntityRegistry


# Every form a caller may use to name entities. `normalize_selection` collapses all of
# them to a plain entity list so downstream code has a single shape to handle.
EntitySelection = Union[
    EntityBase,
    EntitySelector,
    EntityList,
    EntityRegistryView,
    List[Union[EntityBase, EntitySelector]],
]


# Which concrete entity types each selector target class can reach, inverted so an
# entity can be traced back to the target class that selects it.
_ENTITY_TYPE_TO_TARGET_CLASS: Dict[str, str] = {
    entity_type: target_class
    for target_class, entity_types in DEFAULT_TARGET_CLASS_EXPANSION_MAP.items()
    for entity_type in entity_types
}


@dataclass
class SelectorWrapper:
    """Minimal entity-list stand-in that carries selectors only.

    ``expand_entity_list_selectors`` works on anything exposing ``selectors`` and
    ``stored_entities``; this lets loose selectors be expanded without building a
    typed ``EntityList``.
    """

    selectors: List[EntitySelector]


def _selector_class_name(target_class: str) -> str:
    """Name the selector class a user should reach for, given a target class.

    The mapping is not mechanical (``VolumeSelector`` targets ``GenericVolume``), so read
    it back off the selector subclasses instead of guessing from the target class name.
    Only runs when raising, so the subclass scan is not on any hot path.
    """
    for subclass in EntitySelector.__subclasses__():
        if getattr(subclass.model_fields.get("target_class"), "default", None) == target_class:
            return subclass.__name__
    return f"a selector targeting {target_class}"


def _target_classes_named_by(selection: EntitySelection) -> set[str]:
    """Collect the selector target classes a selection refers to.

    Reads the selection rather than its matches so a selector that happens to match
    nothing still identifies what it was looking for.
    """
    if isinstance(selection, EntitySelector):
        return {selection.target_class}

    entity_type_name = None
    if isinstance(selection, EntityRegistryView):
        entity_type_name = selection._entity_type.__name__  # pylint: disable=protected-access
    elif isinstance(selection, EntityBase):
        entity_type_name = selection.private_attribute_entity_type_name
    if entity_type_name is not None:
        target_class = _ENTITY_TYPE_TO_TARGET_CLASS.get(entity_type_name)
        return {target_class} if target_class else set()

    members = []
    if isinstance(selection, EntityList):
        members = list(selection.stored_entities or []) + list(selection.selectors or [])
    elif isinstance(selection, list):
        members = selection
    return set().union(*(_target_classes_named_by(item) for item in members)) if members else set()


def infer_target_class(selection: EntitySelection, *, operation: str) -> str:
    """Return the single selector target class a selection refers to.

    Raises when the selection names no selectable type or more than one, since a caller
    comparing against a target class has no defensible choice in either case.
    """
    target_classes = _target_classes_named_by(selection)
    if not target_classes:
        raise Flow360ValueError(
            f"{operation} cannot tell which kind of entity to compare against. Pass a "
            "non-empty selection holding entities or selectors of a selectable type "
            f"({', '.join(sorted(DEFAULT_TARGET_CLASS_EXPANSION_MAP))})."
        )
    if len(target_classes) > 1:
        raise Flow360ValueError(
            f"{operation} requires a selection of a single entity kind, but got "
            f"{', '.join(sorted(target_classes))}. Call it once per kind."
        )
    return target_classes.pop()


def normalize_selection(
    registry: EntityRegistry,
    selection: EntitySelection,
    *,
    operation: str,
    expected_entity_type: Optional[type] = None,
) -> List[Any]:
    """Collapse any accepted entity-selection form into a deduplicated entity list.

    Accepts a single entity, a single ``EntitySelector``, a list mixing the two, an
    ``EntityList``, or an ``EntityRegistryView``. Callers therefore never branch on the
    input shape. Explicit entities keep their given order and precede selector matches;
    an entity named more than once -- listed twice, or matched by several selectors --
    is kept once.

    Parameters
    ----------
    registry :
        Registry the selectors are expanded against.
    operation :
        User-facing name of the calling operation, used in error messages.
    expected_entity_type :
        When given, selectors and registry views aimed at a different entity type are
        rejected up front. Leave as ``None`` to accept any target class.
    """

    def _check_selector(selector: EntitySelector) -> EntitySelector:
        if expected_entity_type is None or selector.target_class == expected_entity_type.__name__:
            return selector
        raise Flow360ValueError(
            f"{operation} requires a {_selector_class_name(expected_entity_type.__name__)}, "
            f"got selector with target_class='{selector.target_class}'."
        )

    def _expand(selectors: List[EntitySelector]) -> List[Any]:
        return expand_entity_list_selectors(
            registry=registry,
            entity_list=SelectorWrapper(selectors=[_check_selector(item) for item in selectors]),
        )

    if isinstance(selection, EntitySelector):
        resolved = _expand([selection])
    elif isinstance(selection, EntityList):
        resolved = expand_entity_list_selectors(registry=registry, entity_list=selection)
    elif isinstance(selection, EntityRegistryView):
        # pylint: disable=protected-access
        if expected_entity_type is not None and not issubclass(
            selection._entity_type, expected_entity_type
        ):
            raise Flow360ValueError(
                f"{operation} requires a {expected_entity_type.__name__} view, "
                f"got EntityRegistryView of {selection._entity_type.__name__}."
            )
        resolved = list(selection)
    elif isinstance(selection, EntityBase):
        resolved = [selection]
    elif isinstance(selection, list):
        selectors = [item for item in selection if isinstance(item, EntitySelector)]
        entities = [item for item in selection if not isinstance(item, EntitySelector)]
        resolved = entities + (_expand(selectors) if selectors else [])
    else:
        raise Flow360ValueError(
            f"{operation} expected an entity, a selector, a list mixing the two, an "
            f"EntityList, or an EntityRegistryView, got {type(selection).__name__}."
        )

    return deduplicate_entities(resolved)


def expand_entity_list_in_context(
    entity_list,
    params,
    *,
    return_names: bool = False,
) -> Union[List[Any], List[str]]:  # List[EntityBase] | List[str]
    """
    Expand selectors for a deserialized EntityList within the context of SimulationParams.

    Parameters
    ----------
    entity_list :
        EntityList instance that may contain already materialized entities and/or selectors.
    params :
        SimulationParams instance providing the project entity info and selector cache.
    return_names : bool, default False
        When True, return only a list of entity names instead of entity instances.

    Returns
    -------
    list
        List of EntityBase objects or their names depending on `return_names`.
    """

    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    selectors = list(getattr(entity_list, "selectors", []) or [])
    if selectors and asset_cache is None:
        raise Flow360ValueError("The given `params` does not contain any info on usable entities.")

    registry = get_registry_from_asset_cache(asset_cache) if selectors else None
    return expand_entity_list_with_registry(
        entity_list,
        registry,
        return_names=return_names,
    )


def get_registry_from_params(params) -> EntityRegistry:
    """
    Create an EntityRegistry from SimulationParams.

    Parameters
    ----------
    params :
        SimulationParams (or compatible object) that holds `private_attribute_asset_cache`.

    Returns
    -------
    EntityRegistry
        Registry containing all entities from the params.
    """
    if params is None:
        raise ValueError("[Internal] SimulationParams is required to build entity registry.")

    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    if asset_cache is None:
        raise ValueError(
            "[Internal] SimulationParams.private_attribute_asset_cache is required to build entity registry."
        )

    return get_registry_from_asset_cache(asset_cache)


def expand_all_entity_lists_in_place(
    params,
    *,
    merge_mode: Literal["merge", "replace"] = "merge",
    expansion_map: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Expand selectors for all EntityList objects under params in-place.

    This is intended for translation-time expansion where mutating the params object is safe.

    Parameters:
        expansion_map: Optional type expansion mapping for selectors.
    """
    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    entity_info = getattr(asset_cache, "project_entity_info", None)
    if asset_cache is None or entity_info is None:
        # Unit tests may not provide entity_info; in that case selector expansion is not possible.
        return

    expand_all_entity_lists_with_registry_in_place(
        params,
        registry=get_registry_from_asset_cache(asset_cache),
        merge_mode=merge_mode,
        expansion_map=expansion_map,
    )


__all__ = [
    "SelectorWrapper",
    "_register_mirror_entities_in_registry",
    "expand_all_entity_lists_in_place",
    "expand_all_entity_lists_with_registry_in_place",
    "expand_entity_list_in_context",
    "expand_entity_list_with_registry",
    "get_entity_info_and_registry_from_asset_cache",
    "get_entity_info_and_registry_from_dict",
    "get_registry_from_asset_cache",
    "get_registry_from_params",
    "normalize_selection",
]
