"""Entity list expansion helpers shared across results and user utilities."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, List, Union

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_in_place,
)
from flow360.exceptions import Flow360ValueError

if TYPE_CHECKING:
    from flow360.component.simulation.framework.entity_registry import EntityRegistry


def _serialize_selector(selector: Any):
    if isinstance(selector, str):
        return selector
    if hasattr(selector, "model_dump"):
        return selector.model_dump(mode="json", exclude_none=True)
    return copy.deepcopy(selector)


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
        When True, return a list of entity names instead of entity instances.

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
                "The given `params` does not contain any info on usable entities. Please try using "
            )

        wrapper_key = "__entity_list__"
        params_payload = {
            "private_attribute_asset_cache": asset_cache,
            wrapper_key: {
                "stored_entities": stored_entities,
                "selectors": [_serialize_selector(selector) for selector in selectors],
            },
        }
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.framework.entity_selector import (
            expand_entity_selectors_in_place,
        )

        registry = get_registry_from_params(params)
        expand_entity_selectors_in_place(registry, params_payload, merge_mode="merge")
        stored_entities = params_payload[wrapper_key].get("stored_entities", [])

    if not stored_entities:
        return []

    if not all(hasattr(entity, "name") for entity in stored_entities):
        wrapper = {"stored_entities": stored_entities}
        materialize_entities_in_place(wrapper)
        stored_entities = wrapper.get("stored_entities", [])

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
