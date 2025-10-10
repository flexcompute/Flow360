"""Utility functions for the simulation services."""

from collections import deque
from typing import Any


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
