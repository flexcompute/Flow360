"""Entity list expansion helpers shared across results and user utilities."""

# pylint: disable=unused-import
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from flow360_schema.framework.entity.entity_expansion_utils import (  # noqa: F401
    _register_mirror_entities_in_registry,
    expand_all_entity_lists_with_registry_in_place,
    expand_entity_list_with_registry,
    get_entity_info_and_registry_from_asset_cache,
    get_entity_info_and_registry_from_dict,
    get_registry_from_asset_cache,
)

from flow360.exceptions import Flow360ValueError

if TYPE_CHECKING:
    from flow360.component.simulation.framework.entity_registry import EntityRegistry


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
    "_register_mirror_entities_in_registry",
    "expand_all_entity_lists_in_place",
    "expand_all_entity_lists_with_registry_in_place",
    "expand_entity_list_in_context",
    "expand_entity_list_with_registry",
    "get_entity_info_and_registry_from_asset_cache",
    "get_entity_info_and_registry_from_dict",
    "get_registry_from_asset_cache",
    "get_registry_from_params",
]
