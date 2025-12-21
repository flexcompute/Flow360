"""Entity list expansion helpers shared across results and user utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)
from flow360.component.simulation.framework.entity_utils import (
    walk_object_tree_with_cycle_detection,
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

    stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
    selectors = list(getattr(entity_list, "selectors", []) or [])

    if selectors:
        asset_cache = getattr(params, "private_attribute_asset_cache", None)
        if asset_cache is None:
            raise Flow360ValueError(
                "The given `params` does not contain any info on usable entities."
            )

        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.framework.entity_selector import (
            expand_entity_list_selectors,
        )

        registry = get_registry_from_params(params)
        stored_entities = expand_entity_list_selectors(
            registry,
            entity_list,
            selector_cache={},
            merge_mode="merge",
        )

    if not stored_entities:
        return []

    if not all(hasattr(entity, "name") for entity in stored_entities):
        wrapper = {"stored_entities": stored_entities}
        materialize_entities_and_selectors_in_place(wrapper)
        stored_entities = wrapper.get("stored_entities", [])

    # Trigger field validator to filter invalid entity types
    # This ensures consistency with the centralized filtering architecture
    if stored_entities:
        try:
            # Use model_validate to trigger field validator which filters by type
            validated_list = entity_list.__class__.model_validate(
                {"stored_entities": stored_entities}
            )
            stored_entities = validated_list.stored_entities
        except pd.ValidationError:  # pylint: disable=broad-exception-caught
            # If validation fails, fall back to unfiltered list for backward compatibility
            pass

    if return_names:
        return [entity.name for entity in stored_entities]
    return stored_entities


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
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.framework.entity_registry import EntityRegistry

    if params is None:
        raise ValueError("[Internal] SimulationParams is required to build entity registry.")

    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    if asset_cache is None:
        raise ValueError(
            "[Internal] SimulationParams.private_attribute_asset_cache is required to build entity registry."
        )

    entity_info = getattr(asset_cache, "project_entity_info", None)
    if entity_info is None:
        raise ValueError("[Internal] SimulationParams is missing project_entity_info.")

    return EntityRegistry.from_entity_info(entity_info)


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
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.framework.entity_base import EntityList
    from flow360.component.simulation.framework.entity_selector import (
        expand_entity_list_selectors_in_place,
    )

    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    entity_info = getattr(asset_cache, "project_entity_info", None)
    if asset_cache is None or entity_info is None:
        # Unit tests may not provide entity_info; in that case selector expansion is not possible.
        return

    registry = get_registry_from_params(params)
    selector_cache: dict = {}

    def _process_entity_list(obj):
        """Process EntityList objects by expanding their selectors."""
        if isinstance(obj, EntityList):
            expand_entity_list_selectors_in_place(
                registry,
                obj,
                selector_cache=selector_cache,
                merge_mode=merge_mode,
                expansion_map=expansion_map,
            )
            return False  # Don't traverse into EntityList internals
        return True  # Continue traversing other objects

    walk_object_tree_with_cycle_detection(params, _process_entity_list, check_dict=True)


def get_entity_info_and_registry_from_dict(params_as_dict: dict) -> tuple:
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
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.framework.entity_registry import EntityRegistry

    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    if asset_cache is None:
        raise ValueError("[Internal] private_attribute_asset_cache not found in params_as_dict.")

    entity_info_dict = asset_cache.get("project_entity_info")
    if entity_info_dict is None:
        raise ValueError("[Internal] project_entity_info not found in asset cache.")

    # Deserialize entity_info dict to the appropriate EntityInfo class
    from flow360.component.simulation.entity_info import parse_entity_info_model

    entity_info = parse_entity_info_model(entity_info_dict)
    registry = EntityRegistry.from_entity_info(entity_info)

    return entity_info, registry
