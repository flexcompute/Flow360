"""Utilities shared by simulation service and serialization flows."""

from __future__ import annotations

from typing import Any

from flow360_schema.framework.entity.entity_expansion_utils import (
    get_registry_from_asset_cache,
)
from flow360_schema.framework.entity.entity_selector import _process_selectors
from flow360_schema.framework.entity.entity_utils import (
    walk_object_tree_with_cycle_detection,
)

_MIRRORED_ENTITY_TYPE_NAMES = ("MirroredSurface", "MirroredGeometryBodyGroup")


def strip_implicit_edge_split_layers_inplace(params, params_dict: dict) -> dict:
    """
    Remove implicitly injected ``edge_split_layers`` from serialized params.

    This keeps explicitly provided values, including an explicit value equal to the default.
    """
    meshing = getattr(params, "meshing", None)
    defaults = getattr(meshing, "defaults", None)
    if defaults is None:
        return params_dict

    if "edge_split_layers" in defaults.model_fields_set:
        return params_dict

    meshing_dict = params_dict.get("meshing")
    if not isinstance(meshing_dict, dict):
        return params_dict

    defaults_dict = meshing_dict.get("defaults")
    if not isinstance(defaults_dict, dict):
        return params_dict

    defaults_dict.pop("edge_split_layers", None)
    return params_dict


def strip_selector_matches_and_broken_entities_inplace(params) -> Any:
    """
    Strip selector-implied entities and invalid mirrored entities from ``stored_entities`` lists.
    """
    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    project_entity_info = getattr(asset_cache, "project_entity_info", None)
    if asset_cache is None or project_entity_info is None:
        return params

    selector_cache: dict = {}
    registry = get_registry_from_asset_cache(asset_cache)

    valid_mirrored_registry_keys = {
        (entity.private_attribute_entity_type_name, entity.private_attribute_id)
        for entity in registry.find_by_type_name(list(_MIRRORED_ENTITY_TYPE_NAMES))
    }

    def _extract_entity_key(item) -> tuple:
        entity_type = getattr(item, "private_attribute_entity_type_name", None)
        entity_id = getattr(item, "private_attribute_id", None)
        return (entity_type, entity_id)

    def _matched_keyset_for_selectors(selectors_list: list) -> set[tuple]:
        additions_by_class, _ = _process_selectors(
            registry,
            selectors_list,
            selector_cache,
        )
        keys: set = set()
        for items in additions_by_class.values():
            for entity in items:
                keys.add(_extract_entity_key(entity))
        return keys

    def _strip_selector_matches_and_broken_entities(obj) -> bool:
        selectors_list = getattr(obj, "selectors", None)
        stored_entities = getattr(obj, "stored_entities", None)
        if not isinstance(stored_entities, list):
            return True

        if not stored_entities:
            obj.stored_entities = []
            return False

        updated_entities = stored_entities

        if isinstance(selectors_list, list) and selectors_list:
            matched_keys = _matched_keyset_for_selectors(selectors_list)
            if matched_keys:
                updated_entities = [item for item in updated_entities if _extract_entity_key(item) not in matched_keys]
            if not updated_entities:
                obj.stored_entities = []
                return False

        cleaned_entities = []
        for item in updated_entities:
            entity_type, entity_id = _extract_entity_key(item)
            if entity_type is None or entity_id is None:
                cleaned_entities.append(item)
                continue

            if entity_type in _MIRRORED_ENTITY_TYPE_NAMES:
                if (entity_type, entity_id) in valid_mirrored_registry_keys:
                    cleaned_entities.append(item)
                continue

            cleaned_entities.append(item)

        obj.stored_entities = cleaned_entities
        return False

    walk_object_tree_with_cycle_detection(params, _strip_selector_matches_and_broken_entities, check_dict=False)
    return params


__all__ = [
    "strip_implicit_edge_split_layers_inplace",
    "strip_selector_matches_and_broken_entities_inplace",
]
