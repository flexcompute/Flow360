"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from fnmatch import translate as fnmatch_translate
from functools import lru_cache
from typing import Any, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel

# These corresponds to the private_attribute_entity_type_name of supported entity types.
TargetClass = Literal["Surface", "Edge", "GenericVolume", "GeometryBodyGroup"]


class Predicate(Flow360BaseModel):
    """
    Single predicate in a selector.
    """

    # For now only name matching is supported
    attribute: Literal["name"] = pd.Field("name", description="The attribute to match/filter on.")
    operator: Literal[
        "equals",
        "notEquals",
        "in",
        "notIn",
        "matches",
        "notMatches",
    ] = pd.Field()
    value: Union[str, List[str]] = pd.Field()
    # Applies only to matches/notMatches; default to glob if not specified explicitly.
    non_glob_syntax: Optional[Literal["regex"]] = pd.Field(
        None,
        description="If specified, the pattern (`value`) will be treated "
        "as a non-glob pattern with the specified syntax.",
    )


class EntitySelector(Flow360BaseModel):
    """Entity selector for an EntityList.

    - target_class chooses the entity pool
    - logic combines child predicates (AND = intersection, OR = union)
    """

    target_class: TargetClass = pd.Field()
    logic: Literal["AND", "OR"] = pd.Field("AND")
    children: List[Predicate] = pd.Field()


@dataclass
class EntityDictDatabase:
    """
    [Internal Use Only]
    Entity database for entity selectors.
    This is intended to strip off differences between root resources and
    ensure the expansion has a uniform data interface.

    Each data member maps between attribute used for matching and the entity raw JSON dictionary.
    """

    surfaces: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    generic_volumes: list[dict] = field(default_factory=list)
    geometry_body_groups: list[dict] = field(default_factory=list)


def _get_entity_pool(entity_database: EntityDictDatabase, target_class: TargetClass) -> list[dict]:
    """Return the correct entity list from the database for the target class."""
    if target_class == "Surface":
        return entity_database.surfaces
    if target_class == "Edge":
        return entity_database.edges
    if target_class == "GenericVolume":
        return entity_database.generic_volumes
    if target_class == "GeometryBodyGroup":
        return entity_database.geometry_body_groups
    raise ValueError(f"Unknown target class: {target_class}")


@lru_cache(maxsize=2048)
def _compile_regex_cached(pattern: str) -> re.Pattern:
    return re.compile(pattern)


@lru_cache(maxsize=2048)
def _compile_glob_cached(pattern: str) -> re.Pattern:
    return re.compile(fnmatch_translate(pattern))


def _build_name_matcher(predicate: dict):
    """Build a fast predicate(name:str)->bool matcher.

    Precompiles regex/glob and converts membership lists to sets for speed.
    """
    operator = predicate.get("operator")
    value = predicate.get("value")
    non_glob_syntax = predicate.get("non_glob_syntax")

    negate = False
    if operator in ("notEquals", "notIn", "notMatches"):
        negate = True
        base_operator = {
            "notEquals": "equals",
            "notIn": "in",
            "notMatches": "matches",
        }.get(operator)
    else:
        base_operator = operator

    if base_operator == "equals":
        target = value

        def base_match(name: str) -> bool:
            return name == target

    elif base_operator == "in":
        values = set(value or [])

        def base_match(name: str) -> bool:
            return name in values

    elif base_operator == "matches":
        if non_glob_syntax == "regex":
            pattern = _compile_regex_cached(value)
        else:
            pattern = _compile_glob_cached(value)

        def base_match(name: str) -> bool:
            return pattern.fullmatch(name) is not None

    else:

        def base_match(_name: str) -> bool:
            return False

    if negate:
        return lambda name: not base_match(name)
    return base_match


def _build_name_index(pool: list[dict]) -> dict[str, list[int]]:
    name_to_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(pool):
        nm = item.get("name")
        name_to_indices.setdefault(nm, []).append(idx)
    return name_to_indices


def _apply_or_selector(
    pool: list[dict], ordered_children: list[dict], name_to_indices: dict[str, list[int]]
) -> list[dict]:
    indices: set[int] = set()
    for predicate in ordered_children:
        operator = predicate.get("operator")
        if operator == "equals":
            indices.update(name_to_indices.get(predicate.get("value"), []))
            if len(indices) >= len(pool):
                break
            continue
        if operator == "in":
            for v in predicate.get("value") or []:
                indices.update(name_to_indices.get(v, []))
            if len(indices) >= len(pool):
                break
            continue
        matcher = _build_name_matcher(predicate)
        for i, item in enumerate(pool):
            if i in indices:
                continue
            nm = item.get("name")
            if matcher(nm):
                indices.add(i)
        if len(indices) >= len(pool):
            break
    if len(indices) * 4 < len(pool):
        return [pool[i] for i in sorted(indices)]
    return [pool[i] for i in range(len(pool)) if i in indices]


def _apply_and_selector(
    pool: list[dict], ordered_children: list[dict], name_to_indices: dict[str, list[int]]
) -> list[dict]:
    candidate_indices: Optional[set[int]] = None

    def _matched_indices_for_predicate(
        predicate: dict, current_candidates: Optional[set[int]]
    ) -> set[int]:
        operator = predicate.get("operator")
        if operator == "equals":
            return set(name_to_indices.get(predicate.get("value"), []))
        if operator == "in":
            result: set[int] = set()
            for v in predicate.get("value") or []:
                result.update(name_to_indices.get(v, []))
            return result
        matcher = _build_name_matcher(predicate)
        matched: set[int] = set()
        if current_candidates is None:
            for i, item in enumerate(pool):
                nm = item.get("name")
                if matcher(nm):
                    matched.add(i)
            return matched
        for i in current_candidates:
            nm = pool[i].get("name")
            if matcher(nm):
                matched.add(i)
        return matched

    for predicate in ordered_children:
        matched = _matched_indices_for_predicate(predicate, candidate_indices)
        candidate_indices = (
            matched if candidate_indices is None else candidate_indices.intersection(matched)
        )
        if not candidate_indices:
            return []

    assert candidate_indices is not None
    if len(candidate_indices) * 4 < len(pool):
        return [pool[i] for i in sorted(candidate_indices)]
    return [pool[i] for i in range(len(pool)) if i in candidate_indices]


def _apply_single_selector(pool: list[dict], selector_dict: dict) -> list[dict]:
    """Apply one selector over a pool of entity dicts.

    Implementation notes for future readers:
    - We assume selector_dict conforms to the EntitySelector schema (no validation here for speed).
    - We respect the default of logic="AND" when absent.
    - For performance:
      * Reorder predicates under AND so that cheap/selective operations run first.
      * Build a name->indices index to accelerate equals/in where beneficial.
      * Precompile regex/glob matchers once per predicate.
      * Short-circuit when the candidate set becomes empty.
    - Result ordering is stable (by original pool index) to keep the operation idempotent.
    """
    logic = selector_dict.get("logic", "AND")
    children = selector_dict.get("children") or []

    # Fast path: empty predicates -> return nothing. Empty children is actually misuse.
    if not children:
        return []

    # Predicate ordering (AND only): cheap/selective first
    def _cost(predicate: dict) -> int:
        op = predicate.get("operator")
        order = {
            "equals": 0,
            "in": 1,
            "matches": 2,
            "notEquals": 3,
            "notIn": 4,
            "notMatches": 5,
        }
        return order.get(op, 10)

    ordered_children = children if logic == "OR" else sorted(children, key=_cost)

    # Optional name index for equals/in
    need_index = any(p.get("operator") in ("equals", "in") for p in ordered_children)
    name_to_indices: dict[str, list[int]] = _build_name_index(pool) if need_index else {}

    if logic == "OR":
        return _apply_or_selector(pool, ordered_children, name_to_indices)

    return _apply_and_selector(pool, ordered_children, name_to_indices)


def _expand_node_selectors(entity_database: EntityDictDatabase, node: dict) -> None:
    selectors_value = node.get("selectors")
    if not (isinstance(selectors_value, list) and len(selectors_value) > 0):
        return

    additions_by_class: dict[str, list[dict]] = {}
    ordered_target_classes: list[str] = []

    for selector_dict in selectors_value:
        if not isinstance(selector_dict, dict):
            continue
        target_class = selector_dict.get("target_class")
        pool = _get_entity_pool(entity_database, target_class)
        if not pool:
            continue
        if target_class not in additions_by_class:
            additions_by_class[target_class] = []
            ordered_target_classes.append(target_class)
        additions_by_class[target_class].extend(_apply_single_selector(pool, selector_dict))

    existing = node.get("stored_entities")
    base_entities: list[dict] = []
    classes_to_update = set(ordered_target_classes)
    if isinstance(existing, list):
        for item in existing:
            etype = item.get("private_attribute_entity_type_name")
            if etype in classes_to_update:
                continue
            base_entities.append(item)

    for target_class in ordered_target_classes:
        base_entities.extend(additions_by_class.get(target_class, []))

    node["stored_entities"] = base_entities
    node["selectors"] = []


def expand_entity_selectors_in_place(
    entity_database: EntityDictDatabase, params_as_dict: dict
) -> dict:
    """Traverse params_as_dict and expand any EntitySelector in place."""
    queue: deque[Any] = deque([params_as_dict])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            _expand_node_selectors(entity_database, node)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    return params_as_dict
