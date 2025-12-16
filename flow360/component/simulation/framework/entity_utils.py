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
