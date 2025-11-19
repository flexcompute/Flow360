"""Utility functions for the simulation services."""

from collections import deque
from typing import Any

from flow360.component.simulation.entity_info import get_entity_database_for_selectors
from flow360.component.simulation.framework.entity_materializer import (
    _stable_entity_key_from_dict,
)
from flow360.component.simulation.framework.entity_selector import _process_selectors


def has_any_entity_selectors(params_as_dict: dict) -> bool:
    """Return True if there is at least one EntitySelector to expand in params_as_dict.

    This function performs a fast, non-recursive traversal (using a deque) over the
    provided dictionary structure and short-circuits on the first occurrence of a
    potential `EntityList`-like node with a non-empty `selectors` array.

    A node is treated as having selectors if all of the following hold:
      - It is a dict containing the key `selectors`
      - The value for `selectors` is a non-empty list
      - The first element is a dict with at least keys `target_class` and `children`

    Parameters
    ----------
    params_as_dict: dict
        The simulation parameters as a plain dictionary.

    Returns
    -------
    bool
        True if at least one `EntitySelector`-like structure is present; otherwise False.
    """

    if not isinstance(params_as_dict, dict):
        return False

    queue: deque[Any] = deque([params_as_dict])

    while queue:
        node = queue.popleft()

        if isinstance(node, dict):
            # Quick structural check for selectors
            selectors = node.get("selectors")
            if isinstance(selectors, list) and len(selectors) > 0:
                first = selectors[0]
                if isinstance(first, str):
                    # Tokens
                    return True
                if isinstance(first, dict) and "target_class" in first and "children" in first:
                    return True

            # Enqueue children
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    queue.append(value)

        elif isinstance(node, (list, tuple)):
            for item in node:
                if isinstance(item, (dict, list, tuple)):
                    queue.append(item)

    return False


def strip_selector_matches_inplace(params_as_dict: dict) -> dict:
    """
    Remove entities matched by selectors from each EntityList node's stored_entities, in place.

    Rationale:
    - Keep user hand-picked entities distinguishable for the UI by stripping items that are
      implied by EntitySelector expansion.
    - Do not modify schema; operate at dict level without mutating model structure.

    Behavior:
    - For every dict node that has a non-empty `selectors` list, compute the set of additions
      implied by those selectors over the current entity database, and remove those additions
      from the node's `stored_entities` list.
    - Nodes without `selectors` are left untouched.

    Returns the same dict object for chaining.
    """
    if not isinstance(params_as_dict, dict):
        return params_as_dict

    if not has_any_entity_selectors(params_as_dict):
        return params_as_dict

    entity_database = get_entity_database_for_selectors(params_as_dict)
    selector_cache: dict = {}

    known_selectors = {}
    asset_cache = params_as_dict.get("private_attribute_asset_cache", {})
    if isinstance(asset_cache, dict):
        selectors_list = asset_cache.get("selectors")
        if isinstance(selectors_list, list):
            for s in selectors_list:
                if isinstance(s, dict) and "name" in s:
                    known_selectors[s["name"]] = s

    def _matched_keyset_for_selectors(selectors_value: list) -> set:
        additions_by_class, _ = _process_selectors(
            entity_database, selectors_value, selector_cache, known_selectors=known_selectors
        )
        keys: set = set()
        for items in additions_by_class.values():
            for d in items:
                if isinstance(d, dict):
                    keys.add(_stable_entity_key_from_dict(d))
        return keys

    def _visit_dict(node: dict) -> None:
        selectors_value = node.get("selectors")
        if isinstance(selectors_value, list) and len(selectors_value) > 0:
            matched_keys = _matched_keyset_for_selectors(selectors_value)
            se = node.get("stored_entities")
            if isinstance(se, list) and len(se) > 0:
                node["stored_entities"] = [
                    item
                    for item in se
                    if not (
                        isinstance(item, dict)
                        and _stable_entity_key_from_dict(item) in matched_keys
                    )
                ]
        for v in node.values():
            if isinstance(v, (dict, list)):
                _visit(v)

    def _visit(node):
        if isinstance(node, dict):
            _visit_dict(node)
            return
        if isinstance(node, list):
            for it in node:
                if isinstance(it, (dict, list)):
                    _visit(it)

    _visit(params_as_dict)
    return params_as_dict
