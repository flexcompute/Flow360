"""Utility functions for the simulation services."""

from typing import Any

from flow360.component.simulation.framework.entity_expansion_utils import (
    get_registry_from_params,
)
from flow360.component.simulation.framework.entity_selector import _process_selectors


def strip_selector_matches_inplace(params) -> Any:
    """
    Remove entities matched by selectors from each EntityList's stored_entities, in place.

    Rationale:
    - Keep user hand-picked entities distinguishable for the UI by stripping items that are
      implied by EntitySelector expansion.
    - Operate on the deserialized params object to avoid dict-level selector handling.

    Behavior:
    - For every EntityList-like object that has a non-empty `selectors` list, compute the set
      of entities implied by those selectors over the registry, and remove those implied entities
      from its `stored_entities` list.

    Returns the same object for chaining.
    """
    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    project_entity_info = getattr(asset_cache, "project_entity_info", None)
    if asset_cache is None or project_entity_info is None:
        # Compatibility with some unit tests.
        return params

    selector_cache: dict = {}
    registry = get_registry_from_params(params)

    def _extract_entity_key(item) -> tuple:
        """Extract stable key from entity object."""
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

    visited: set[int] = set()

    def _visit(obj) -> None:
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        selectors_list = getattr(obj, "selectors", None)
        stored_entities = getattr(obj, "stored_entities", None)
        if isinstance(selectors_list, list) and selectors_list:
            if isinstance(stored_entities, list) and stored_entities:
                matched_keys = _matched_keyset_for_selectors(selectors_list)
                obj.stored_entities = [
                    item
                    for item in stored_entities
                    if _extract_entity_key(item) not in matched_keys
                ]
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, (list, tuple)) or hasattr(item, "__dict__"):
                    _visit(item)
            return

        if hasattr(obj, "__dict__"):
            for field_value in obj.__dict__.values():
                if isinstance(field_value, (list, tuple)) or hasattr(field_value, "__dict__"):
                    _visit(field_value)

    _visit(params)
    return params
