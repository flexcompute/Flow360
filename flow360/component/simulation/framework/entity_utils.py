"""Shared utilities for entity operations."""

import hashlib
import json
import re
import uuid
from functools import lru_cache
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

# Define a default set of types that should not be merged/deduplicated.
DEFAULT_NOT_MERGED_TYPES = frozenset({"Point"})


def generate_uuid():
    """generate a unique identifier for non-persistent entities. Required by front end."""
    return str(uuid.uuid4())


def get_entity_type(item: Any) -> Optional[str]:
    """Get entity type name from dict or object."""
    if isinstance(item, dict):
        return item.get("private_attribute_entity_type_name")
    return getattr(item, "private_attribute_entity_type_name", type(item).__name__)


def get_entity_key(item: Any) -> tuple:
    """Return a stable deduplication key for an entity (dict or object).

    Strategy:
    1. (type, private_attribute_id) if ID exists.
    2. For dicts without ID: (type, hash(json_dump(content))).
    3. For objects without ID: (type, id(object)).
    """
    t = get_entity_type(item)

    # Try getting ID
    if isinstance(item, dict):
        pid = item.get("private_attribute_id")
    else:
        pid = getattr(item, "private_attribute_id", None)

    if pid:
        return (t, pid)

    # Fallback
    if isinstance(item, dict):
        # Hash content for dicts without ID
        # Exclude volatile fields
        data = {k: v for k, v in item.items() if k not in ("private_attribute_input_cache",)}
        return (t, hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest())

    # Object identity for objects without ID
    return (t, id(item))


def deduplicate_entities(
    entities: Iterable[Any],
    *,
    processor: Optional[Callable[[Any], Tuple[Any, tuple]]] = None,
    not_merged_types: Set[str] = DEFAULT_NOT_MERGED_TYPES,
) -> List[Any]:
    """
    Process and deduplicate a list of entities.

    Parameters
    ----------
    entities : Iterable[Any]
        Input list of entities (dicts or objects).
    processor : Callable[[Any], Tuple[Any, tuple]], optional
        Function that processes an item and returns (processed_item, key).
        If None, the item is used as is, and key is derived via get_entity_key.
    not_merged_types : Set[str]
        Set of entity type names to skip deduplication (always keep).

    Returns
    -------
    List[Any]
        New list of processed and deduplicated entities.
    """
    new_list = []
    seen = set()

    for item in entities:
        if processor:
            obj, key = processor(item)
        else:
            obj = item
            key = get_entity_key(obj)

        # Check if we should skip deduplication for this type
        t = get_entity_type(obj)
        if t in not_merged_types:
            new_list.append(obj)
            continue

        if key in seen:
            continue

        seen.add(key)
        new_list.append(obj)

    return new_list


def walk_object_tree_with_cycle_detection(
    obj: Any,
    visitor: Callable[[Any], bool],
    *,
    check_dict: bool = True,
) -> None:
    """
    Walk an object tree using depth-first traversal with cycle detection.

    This utility provides a reusable pattern for traversing nested object structures
    (lists, tuples, dicts, and objects with __dict__) while avoiding infinite loops
    caused by circular references.

    Parameters
    ----------
    obj : Any
        The root object to start traversal from
    visitor : Callable[[Any], bool]
        Function called on each object. Should return True to continue traversal
        into this object's children, or False to skip traversal of children.
        The visitor handles all type-specific logic.
    check_dict : bool, default True
        Whether to traverse dict objects. Set to False if you only want to traverse
        list/tuple/object-with-__dict__.

    Notes
    -----
    - Uses id() to track visited objects, preventing revisiting in cycles
    - Traverses list, tuple, dict (if check_dict=True), and objects with __dict__
    - The visitor function controls what happens at each node and whether to recurse

    Examples
    --------
    >>> def print_entity_lists(obj):
    ...     if isinstance(obj, EntityList):
    ...         print(f"Found EntityList: {obj}")
    ...         return False  # Don't traverse into EntityList internals
    ...     return True  # Continue traversing
    >>> walk_object_tree_with_cycle_detection(params, print_entity_lists)
    """
    visited: set[int] = set()

    def _should_traverse(item):
        """Check if an item should be traversed."""
        return (
            isinstance(item, (list, tuple))
            or hasattr(item, "__dict__")
            or (check_dict and isinstance(item, dict))
        )

    def _walk(current_obj):
        obj_id = id(current_obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Call visitor and check if we should continue traversing
        should_continue = visitor(current_obj)
        if not should_continue:
            return

        # Get children to traverse based on object type
        children = []
        if isinstance(current_obj, (list, tuple)):
            children = current_obj
        elif check_dict and isinstance(current_obj, dict):
            children = current_obj.values()
        elif hasattr(current_obj, "__dict__"):
            children = current_obj.__dict__.values()

        # Traverse children
        for child in children:
            if _should_traverse(child):
                _walk(child)

    _walk(obj)


@lru_cache(maxsize=2048)
def compile_glob_cached(pattern: str) -> re.Pattern:
    """Compile an extended-glob pattern via wcmatch to a fullmatch-ready regex.

    We enable extended glob features including brace expansion, extglob groups,
    and globstar. We intentionally avoid PATHNAME semantics because entity
    names are not paths in this context, and we keep case-sensitive matching to
    remain predictable across platforms.
    """
    # Strong requirement: wcmatch must be present to support full glob features.
    # pylint: disable=import-outside-toplevel
    from wcmatch import fnmatch as wfnmatch

    # Enforce case-sensitive matching across platforms (Windows defaults to case-insensitive).
    wc_flags = wfnmatch.BRACE | wfnmatch.EXTMATCH | wfnmatch.DOTMATCH | wfnmatch.CASE
    translated = wfnmatch.translate(pattern, flags=wc_flags)
    # wcmatch.translate may return a tuple: (list_of_regex_strings, list_of_flags)
    if isinstance(translated, tuple):
        regex_parts, _flags = translated
        if isinstance(regex_parts, list) and len(regex_parts) > 1:

            def _strip_anchors(expr: str) -> str:
                if expr.startswith("^"):
                    expr = expr[1:]
                if expr.endswith("$"):
                    expr = expr[:-1]
                return expr

            stripped = [_strip_anchors(s) for s in regex_parts]
            combined = "^(?:" + ")|(?:".join(stripped) + ")$"
            return re.compile(combined)
        if isinstance(regex_parts, list) and len(regex_parts) == 1:
            return re.compile(regex_parts[0])
    # Otherwise, assume it's a single regex string
    return re.compile(translated)
