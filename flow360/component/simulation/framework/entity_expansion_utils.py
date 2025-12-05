"""Entity list expansion helpers shared across results and user utilities."""

from __future__ import annotations

import copy
from typing import Any, List, Union

from flow360.component.simulation.framework.entity_materializer import (
    _stable_entity_key_from_obj,
    materialize_entities_in_place,
)
from flow360.component.simulation.framework.entity_selector import SelectorEntityPool
from flow360.exceptions import Flow360ValueError


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

        selector_pool = get_selector_pool_from_params(params, use_instances=True)
        expand_entity_selectors_in_place(selector_pool, params_payload, merge_mode="merge")
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


def _node_get(container, attribute, default=None):
    if isinstance(container, dict):
        return container.get(attribute, default)
    return getattr(container, attribute, default)


def _get_grouped_entities_from_geometry(entity_info: Any, entity_type_name: str) -> list:
    """
    Extract entities based on current grouping tag for GeometryEntityInfo.

    Mimics the logic from GeometryEntityInfo._get_list_of_entities.
    """
    if entity_type_name == "face":
        attribute_names = _node_get(entity_info, "face_attribute_names", []) or []
        grouped_list = _node_get(entity_info, "grouped_faces", []) or []
        group_tag = _node_get(entity_info, "face_group_tag")
    elif entity_type_name == "edge":
        attribute_names = _node_get(entity_info, "edge_attribute_names", []) or []
        grouped_list = _node_get(entity_info, "grouped_edges", []) or []
        group_tag = _node_get(entity_info, "edge_group_tag")
    elif entity_type_name == "body":
        attribute_names = _node_get(entity_info, "body_attribute_names", []) or []
        grouped_list = _node_get(entity_info, "grouped_bodies", []) or []
        group_tag = _node_get(entity_info, "body_group_tag")
    else:
        return []

    # If no grouping tag is set, use the default (first non-ID tag)
    if group_tag is None:
        if not attribute_names:
            return []
        # Get first non-ID tag (mimics _get_default_grouping_tag logic)
        id_tag = f"{entity_type_name}Id"
        for tag in attribute_names:
            if tag != id_tag:
                group_tag = tag
                break
        if group_tag is None:
            group_tag = id_tag

    # Find the index of the grouping tag in attribute_names
    if group_tag in attribute_names:
        index = attribute_names.index(group_tag)
        if index < len(grouped_list):
            return grouped_list[index]

    return []


def _extract_geometry_entities(entity_info: Any) -> tuple[list, list, list]:
    """Extract entities from GeometryEntityInfo."""
    surfaces = _get_grouped_entities_from_geometry(entity_info, "face")

    edges = []
    if _node_get(entity_info, "edge_ids"):
        edges = _get_grouped_entities_from_geometry(entity_info, "edge")

    geometry_body_groups = []
    if _node_get(entity_info, "body_attribute_names"):
        geometry_body_groups = _get_grouped_entities_from_geometry(entity_info, "body")

    return surfaces, edges, geometry_body_groups


def _extract_volume_mesh_entities(entity_info: Any) -> tuple[list, list]:
    """Extract entities from VolumeMeshEntityInfo."""
    surfaces = _node_get(entity_info, "boundaries", []) or []
    generic_volumes = _node_get(entity_info, "zones", []) or []
    return surfaces, generic_volumes


def _extract_surface_mesh_entities(entity_info: Any) -> list:
    """Extract entities from SurfaceMeshEntityInfo."""
    return _node_get(entity_info, "boundaries", []) or []


def _build_selector_entity_pool_from_entity_info(entity_info: Any) -> SelectorEntityPool:
    """Normalize entity info (dict or object) into SelectorEntityPool for selector operations."""
    entity_info_type = _node_get(entity_info, "type_name")

    surfaces: list = []
    edges: list = []
    generic_volumes: list = []
    geometry_body_groups: list = []

    if entity_info_type == "GeometryEntityInfo":
        surfaces, edges, geometry_body_groups = _extract_geometry_entities(entity_info)
    elif entity_info_type == "VolumeMeshEntityInfo":
        surfaces, generic_volumes = _extract_volume_mesh_entities(entity_info)
    elif entity_info_type == "SurfaceMeshEntityInfo":
        surfaces = _extract_surface_mesh_entities(entity_info)

    return SelectorEntityPool(
        surfaces=surfaces,
        edges=edges,
        generic_volumes=generic_volumes,
        geometry_body_groups=geometry_body_groups,
    )


def get_selector_pool_from_dict(params_as_dict: dict) -> SelectorEntityPool:
    """
    Go through the simulation json and retrieve the entity database for entity selectors.

    This function extracts all entities from private_attribute_asset_cache and converts them
    to dictionary format for use in entity selection operations. For GeometryEntityInfo, it
    respects the current grouping tags (face_group_tag, edge_group_tag, body_group_tag).

    Parameters:
        params_as_dict: Simulation parameters as dictionary containing private_attribute_asset_cache

    Returns:
        SelectorEntityPool: Database containing all available entities as dictionaries
    """
    # Extract and validate asset cache
    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    if asset_cache is None:
        raise ValueError("[Internal] private_attribute_asset_cache not found in params_as_dict.")

    entity_info = asset_cache.get("project_entity_info")
    if entity_info is None:
        raise ValueError("[Internal] project_entity_info not found in asset cache.")

    return _build_selector_entity_pool_from_entity_info(entity_info)


def get_selector_pool_from_params(params, *, use_instances: bool = False) -> SelectorEntityPool:
    """
    Retrieve the entity database for selectors from a SimulationParams instance.

    Parameters
    ----------
    params :
        SimulationParams (or compatible object) that holds `private_attribute_asset_cache`.
    use_instances : bool, default False
        When True, returns databases backed by deserialized entity objects.
        When False, falls back to the JSON/dict path (same as `get_selector_pool_from_dict`).
    """

    if params is None:
        raise ValueError("[Internal] SimulationParams is required to build entity database.")

    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    if asset_cache is None:
        raise ValueError(
            "[Internal] SimulationParams.private_attribute_asset_cache is required to build entity database."
        )

    if not use_instances:
        params_as_dict = params.model_dump(mode="json", exclude_none=True)
        return get_selector_pool_from_dict(params_as_dict)

    entity_info = getattr(asset_cache, "project_entity_info", None)
    if entity_info is None:
        raise ValueError("[Internal] SimulationParams is missing project_entity_info.")

    return _build_selector_entity_pool_from_entity_info(entity_info)


def build_entity_pool_from_entity_info(entity_info: Any) -> dict:
    """
    Build an entity pool from entity_info for use in materialization.

    This function extracts all entities (persistent, draft, and ghost) from the entity_info
    and creates a dict keyed by (type_name, private_attribute_id) for use as a pre-populated
    cache during entity materialization.

    This enables reference identity between entity_info and params.
    And also standarizes access of entities from entity_info.

    Parameters
    ----------
    entity_info : EntityInfoModel
        The entity info containing all entities (GeometryEntityInfo, VolumeMeshEntityInfo,
        or SurfaceMeshEntityInfo).

    Returns
    -------
    dict
        A dict mapping (type_name, private_attribute_id) tuples to entity instances.
    """
    pool = {}

    def _add_entities_to_pool(entities):
        """Helper to add entities to pool."""
        for entity in entities:
            key = _stable_entity_key_from_obj(entity)
            pool[key] = entity

    entity_info_type = _node_get(entity_info, "type_name")

    # Collect persistent entities based on entity_info type
    if entity_info_type == "GeometryEntityInfo":
        for attr_name in ["grouped_faces", "grouped_edges", "grouped_bodies"]:
            for group in _node_get(entity_info, attr_name, []) or []:
                _add_entities_to_pool(group)

    elif entity_info_type == "VolumeMeshEntityInfo":
        for attr_name in ["boundaries", "zones"]:
            _add_entities_to_pool(_node_get(entity_info, attr_name, []) or [])

    elif entity_info_type == "SurfaceMeshEntityInfo":
        _add_entities_to_pool(_node_get(entity_info, "boundaries", []) or [])

    # Collect draft and ghost entities
    for attr_name in ["draft_entities", "ghost_entities"]:
        _add_entities_to_pool(_node_get(entity_info, attr_name, []) or [])

    return pool
