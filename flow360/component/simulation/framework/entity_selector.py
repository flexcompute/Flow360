"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import pydantic as pd
from typing_extensions import Self

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_utils import (
    compile_glob_cached,
    generate_uuid,
)
from flow360.log import log

# These corresponds to the private_attribute_entity_type_name of supported entity types.
TargetClass = Literal["Surface", "Edge", "GenericVolume", "GeometryBodyGroup"]

EntityNode = Union[Any, Dict[str, Any]]  # Union[EntityBase, Dict[str, Any]]


class Predicate(Flow360BaseModel):
    """
    Single predicate in a selector.
    """

    # For now only name matching is supported
    attribute: Literal["name"] = pd.Field("name", description="The attribute to match/filter on.")
    operator: Literal[
        "any_of",
        "not_any_of",
        "matches",
        "not_matches",
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
    description: Optional[str] = pd.Field(
        None, description="Customizable description of the selector."
    )
    selector_id: str = pd.Field(
        default_factory=generate_uuid,
        description="[Internal] Unique identifier for the selector.",
        frozen=True,
    )
    # Unique name for global reuse
    name: str = pd.Field(description="Unique name for this selector.")
    logic: Literal["AND", "OR"] = pd.Field("AND")
    children: List[Predicate] = pd.Field()

    @pd.validate_call
    def match(
        self,
        pattern: str,
        *,
        attribute: Literal["name"] = "name",
        syntax: Literal["glob", "regex"] = "glob",
    ) -> Self:
        """Append a matches predicate and return self for chaining."""
        # pylint: disable=no-member
        self.children.append(
            Predicate(
                attribute=attribute,
                operator="matches",
                value=pattern,
                non_glob_syntax=("regex" if syntax == "regex" else None),
            )
        )
        return self

    @pd.validate_call
    def not_match(
        self,
        pattern: str,
        *,
        attribute: Literal["name"] = "name",
        syntax: Literal["glob", "regex"] = "glob",
    ) -> Self:
        """Append a notMatches predicate and return self for chaining."""
        # pylint: disable=no-member
        self.children.append(
            Predicate(
                attribute=attribute,
                operator="not_matches",
                value=pattern,
                non_glob_syntax=("regex" if syntax == "regex" else None),
            )
        )
        return self

    @pd.validate_call
    def any_of(self, values: List[str], *, attribute: Literal["name"] = "name") -> Self:
        """Append an in predicate and return self for chaining."""
        # pylint: disable=no-member
        self.children.append(Predicate(attribute=attribute, operator="any_of", value=values))
        return self

    @pd.validate_call
    def not_any_of(self, values: List[str], *, attribute: Literal["name"] = "name") -> Self:
        """Append a notIn predicate and return self for chaining."""
        # pylint: disable=no-member
        self.children.append(Predicate(attribute=attribute, operator="not_any_of", value=values))
        return self


@dataclass
class EntityDictDatabase:
    """
    [Internal Use Only]

    Entity database for entity selectors. Provides a unified data interface for entity selectors.

    Stored items can be either plain dictionaries (serialized form) or deserialized entity objects.
    """

    surfaces: list[EntityNode] = field(default_factory=list)
    edges: list[EntityNode] = field(default_factory=list)
    generic_volumes: list[EntityNode] = field(default_factory=list)
    geometry_body_groups: list[EntityNode] = field(default_factory=list)


########## API IMPLEMENTATION ##########


def _validate_selector_factory_common(
    method_name: str,
    *,
    name: str,
    attribute: str,
    logic: str,
    syntax: Optional[str] = None,
) -> None:
    """
    Validate common arguments for SelectorFactory methods.

    This performs friendly, actionable validation with clear error messages.
    """
    # name: required and meaningful
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"SelectorFactory.{method_name}: 'name' must be a non-empty string; "
            "it is the selector's unique identifier."
        )

    # attribute: currently only 'name' is supported
    if attribute != "name":
        raise ValueError(
            f"SelectorFactory.{method_name}: attribute must be 'name'. Other attributes are not supported."
        )

    # logic
    if logic not in ("AND", "OR"):
        raise ValueError(
            f"SelectorFactory.{method_name}: logic must be one of {{'AND','OR'}}. Got: {logic!r}."
        )

    # syntax (if applicable)
    if syntax is not None and syntax not in ("glob", "regex"):
        raise ValueError(
            f"SelectorFactory.{method_name}: syntax must be one of {{'glob','regex'}}. Got: {syntax!r}."
        )


def _validate_selector_pattern(method_name: str, pattern: str) -> None:
    """Validate the pattern argument for match/not_match."""
    if not isinstance(pattern, str) or len(pattern) == 0:
        raise ValueError(f"SelectorFactory.{method_name}: pattern must be a non-empty string.")


def _validate_selector_values(method_name: str, values: list[str]) -> None:
    """Validate values argument for any_of/not_any_of."""
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(
            f"SelectorFactory.{method_name}: values must be a non-empty list of strings."
        )
    for i, v in enumerate(values):
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"SelectorFactory.{method_name}: values[{i}] must be a non-empty string."
            )


class SelectorFactory:
    """
    Mixin providing class-level helpers to build EntitySelector instances with
    preset predicates.
    """

    @classmethod
    # pylint: disable=too-many-arguments
    def match(
        cls,
        pattern: str,
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        syntax: Literal["glob", "regex"] = "glob",
        logic: Literal["AND", "OR"] = "AND",
        description: Optional[str] = None,
    ) -> EntitySelector:
        """
        Create an EntitySelector for this class and seed it with one matches predicate.

        Example
        -------
        >>> # Glob match on Surface names (AND logic by default)
        >>> fl.Surface.match("wing*", name="wing_sel")
        >>> # Regex full match
        >>> fl.Surface.match(r"^wing$", syntax="regex", name="wing_sel")
        >>> # Chain more predicates with AND logic
        >>> fl.Surface.match("wing*", name="wing_sel").not_any_of(["wing"])
        >>> # Use OR logic across predicates (short alias)
        >>> fl.Surface.match("s1", name="s1_or", logic="OR").any_of(["tail"])

        ====
        """
        _validate_selector_factory_common(
            "match", name=name, attribute=attribute, logic=logic, syntax=syntax
        )
        _validate_selector_pattern("match", pattern)

        selector = generate_entity_selector_from_class(
            selector_name=name,
            entity_class=cls,
            logic=logic,
            selector_description=description,
        )
        selector.match(pattern, attribute=attribute, syntax=syntax)
        return selector

    @classmethod
    # pylint: disable=too-many-arguments
    def not_match(
        cls,
        pattern: str,
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        syntax: Literal["glob", "regex"] = "glob",
        logic: Literal["AND", "OR"] = "AND",
        description: Optional[str] = None,
    ) -> EntitySelector:
        """Create an EntitySelector and seed a notMatches predicate.

        Example
        -------
        >>> # Exclude all surfaces ending with '-root'
        >>> fl.Surface.match("*", name="exclude_root").not_match("*-root")
        >>> # Exclude by regex
        >>> fl.Surface.match("*").not_match(r".*-(root|tip)$", syntax="regex")

        ====
        """
        _validate_selector_factory_common(
            "not_match", name=name, attribute=attribute, logic=logic, syntax=syntax
        )
        _validate_selector_pattern("not_match", pattern)

        selector = generate_entity_selector_from_class(
            selector_name=name,
            entity_class=cls,
            logic=logic,
            selector_description=description,
        )
        selector.not_match(pattern, attribute=attribute, syntax=syntax)
        return selector

    # pylint: disable=too-many-arguments
    @classmethod
    def any_of(
        cls,
        values: List[str],
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        logic: Literal["AND", "OR"] = "AND",
        description: Optional[str] = None,
    ) -> EntitySelector:
        """Create an EntitySelector and seed an in predicate.

        Example
        -------
        >>> fl.Surface.any_of(["a", "b", "c"])
        >>> # Equivalent alias
        >>> fl.Surface.in_(["a", "b", "c"])
        >>> # Combine with not_any_of to subtract
        >>> fl.Surface.any_of(["a", "b", "c"]).not_any_of(["b"])

        ====
        """
        _validate_selector_factory_common("any_of", name=name, attribute=attribute, logic=logic)
        _validate_selector_values("any_of", values)

        selector = generate_entity_selector_from_class(
            selector_name=name,
            entity_class=cls,
            logic=logic,
            selector_description=description,
        )
        selector.any_of(values, attribute=attribute)
        return selector

    # pylint: disable=too-many-arguments
    @classmethod
    def not_any_of(
        cls,
        values: List[str],
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        logic: Literal["AND", "OR"] = "AND",
        description: Optional[str] = None,
    ) -> EntitySelector:
        """Create an EntitySelector and seed a notIn predicate.

        Example
        -------
        >>> # Select all except those in the set
        >>> fl.Surface.match("*").not_any_of(["a", "b"])

        ====
        """
        _validate_selector_factory_common("not_any_of", name=name, attribute=attribute, logic=logic)
        _validate_selector_values("not_any_of", values)  # type: ignore[arg-type]

        selector = generate_entity_selector_from_class(
            selector_name=name,
            entity_class=cls,
            logic=logic,
            selector_description=description,
        )
        selector.not_any_of(values, attribute=attribute)
        return selector


def generate_entity_selector_from_class(
    selector_name: str,
    entity_class: type,
    logic: Literal["AND", "OR"] = "AND",
    selector_description: Optional[str] = None,
) -> EntitySelector:
    """
    Create a new selector for the given entity class.

    entity_class should be one of the supported entity types (Surface, Edge, GenericVolume, GeometryBodyGroup).
    """
    class_name = getattr(entity_class, "__name__", str(entity_class))
    allowed_classes = get_args(TargetClass)
    assert (
        class_name in allowed_classes
    ), f"Unknown entity class: {entity_class} for generating entity selector."

    return EntitySelector(
        name=selector_name,
        description=selector_description,
        target_class=class_name,
        logic=logic,
        children=[],
    )


########## EXPANSION IMPLEMENTATION ##########
def _get_entity_pool(
    entity_database: EntityDictDatabase, target_class: TargetClass
) -> list[EntityNode]:
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


def _get_node_attribute(entity: Any, attribute: str):
    """Return attribute value from either dicts or entity objects."""
    if isinstance(entity, dict):
        return entity.get(attribute)
    return getattr(entity, attribute, None)


def _collect_known_selectors_from_asset_cache(asset_cache) -> dict[str, dict]:
    """Normalize selector definitions originating from asset cache."""
    if asset_cache is None:
        return {}
    if isinstance(asset_cache, dict):
        selectors = asset_cache.get("used_selectors", [])
    else:
        selectors = getattr(asset_cache, "used_selectors", [])
    known: dict[str, dict] = {}
    for item in selectors or []:
        if isinstance(item, str):
            continue
        if hasattr(item, "model_dump"):
            selector_dict = item.model_dump(mode="json", exclude_none=True)
        elif isinstance(item, dict):
            selector_dict = item
        else:
            continue
        selector_id = selector_dict.get("selector_id")
        if selector_id:
            known[selector_id] = selector_dict
    return known


def _get_attribute_value(entity: Any, attribute: str) -> Optional[str]:
    """Return the scalar string value of an attribute, or None if absent/unsupported.

    Only scalar string attributes are supported by this matcher layer for now.
    """
    val = _get_node_attribute(entity, attribute)
    if isinstance(val, str):
        return val
    return None


def _build_value_matcher(predicate: dict):
    """
    Build a fast predicate(value: Optional[str])->bool matcher.

    Precompiles regex/glob and converts membership lists to sets for speed.
    """
    operator = predicate.get("operator")
    value = predicate.get("value")
    non_glob_syntax = predicate.get("non_glob_syntax")

    negate = False
    if operator in ("not_any_of", "not_matches"):
        negate = True
        base_operator = {
            "not_any_of": "any_of",
            "not_matches": "matches",
        }.get(operator)
    else:
        base_operator = operator

    if base_operator == "any_of":
        values = set(value or [])

        def base_match(val: Optional[str]) -> bool:
            return val in values

    elif base_operator == "matches":
        if non_glob_syntax == "regex":
            pattern = _compile_regex_cached(value)
        else:
            pattern = compile_glob_cached(value)

        def base_match(val: Optional[str]) -> bool:
            return isinstance(val, str) and (pattern.fullmatch(val) is not None)

    else:

        def base_match(_val: Optional[str]) -> bool:
            return False

    if negate:
        return lambda val: not base_match(val)
    return base_match


def _build_index(pool: list[EntityNode], attribute: str) -> dict[str, list[int]]:
    """Build an index for in lookups on a given attribute."""
    value_to_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(pool):
        val = _get_node_attribute(item, attribute)
        if isinstance(val, str):
            value_to_indices.setdefault(val, []).append(idx)
    return value_to_indices


def _apply_or_selector(
    pool: list[EntityNode],
    ordered_children: list[dict],
) -> list[Any]:
    indices: set[int] = set()
    for predicate in ordered_children:
        attribute = predicate.get("attribute", "name")
        matcher = _build_value_matcher(predicate)
        for i, item in enumerate(pool):
            if i in indices:
                continue
            if matcher(_get_attribute_value(item, attribute)):
                indices.add(i)
        if len(indices) >= len(pool):
            break
    if len(indices) * 4 < len(pool):
        return [pool[i] for i in sorted(indices)]
    return [pool[i] for i in range(len(pool)) if i in indices]


def _apply_and_selector(
    pool: list[EntityNode],
    ordered_children: list[dict],
    indices_by_attribute: dict[str, dict[str, list[int]]],
) -> list[Any]:
    candidate_indices: Optional[set[int]] = None

    def _matched_indices_for_predicate(
        predicate: dict, current_candidates: Optional[set[int]]
    ) -> set[int]:
        operator = predicate.get("operator")
        attribute = predicate.get("attribute", "name")
        if operator == "any_of":
            idx_map = indices_by_attribute.get(attribute)
            if idx_map is not None:
                result: set[int] = set()
                for v in predicate.get("value") or []:
                    result.update(idx_map.get(v, []))
                return result
        matcher = _build_value_matcher(predicate)
        matched: set[int] = set()
        if current_candidates is None:
            for i, item in enumerate(pool):
                if matcher(_get_attribute_value(item, attribute)):
                    matched.add(i)
            return matched
        for i in current_candidates:
            if matcher(_get_attribute_value(pool[i], attribute)):
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


def _apply_single_selector(pool: list[EntityNode], selector_dict: dict) -> list[EntityNode]:
    """Apply one selector over a pool of entities (dicts or objects).

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
            "any_of": 0,
            "matches": 1,
            "not_any_of": 2,
            "not_matches": 3,
        }
        return order.get(op, 10)

    ordered_children = children if logic == "OR" else sorted(children, key=_cost)

    # Optional per-attribute indices for in
    attributes_needing_index = {
        p.get("attribute", "name") for p in ordered_children if p.get("operator") == "any_of"
    }
    indices_by_attribute: dict[str, dict[str, list[int]]] = (
        {attr: _build_index(pool, attr) for attr in attributes_needing_index}
        if attributes_needing_index
        else {}
    )

    if logic == "OR":
        # Favor a full scan for OR to preserve predictable union behavior
        # and avoid over-indexing that could complicate ordering.
        result = _apply_or_selector(pool, ordered_children)
    else:
        result = _apply_and_selector(pool, ordered_children, indices_by_attribute)

    if not result:
        name = selector_dict.get("name", "unnamed")
        target_class = selector_dict.get("target_class", "Unknown")
        log.warning(
            "Entity selector '%s' (target_class=%s) matched 0 entities. "
            "Please check if the entity name or pattern is correct.",
            name,
            target_class,
        )

    return result


def _get_selector_cache_key(selector_dict: dict) -> tuple:
    """
    Return the cache key for a selector: requires unique name.

    We mandate a unique identifier per selector; use ("name", target_class, name)
    for stable global reuse. If neither `name` is provided, fall back to a
    structural key so different unnamed selectors won't collide.
    """
    target_class = selector_dict.get("target_class")
    name = selector_dict.get("name")
    if name:
        return ("name", target_class, name)

    logic = selector_dict.get("logic", "AND")
    children = selector_dict.get("children") or []

    def _normalize_value(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    predicates = tuple(
        (
            p.get("attribute", "name"),
            p.get("operator"),
            _normalize_value(p.get("value")),
            p.get("non_glob_syntax"),
        )
        for p in children
        if isinstance(p, dict)
    )
    return ("struct", target_class, logic, predicates)


def _process_selectors(
    entity_database: EntityDictDatabase,
    selectors_value: list,
    selector_cache: dict,
    known_selectors: dict[str, dict] = None,
) -> tuple[dict[str, list[EntityNode]], list[str]]:
    """Process selectors and return additions grouped by class.

    This function iterates over the list of selectors (which can be full dictionaries or
    string tokens).
    - If a selector is a string token, it looks up the full definition in `known_selectors`.
    - If a selector is a dictionary, it uses it directly.
    - It then applies the selector logic to find matching entities from the database.
    - Results are cached in `selector_cache` to avoid re-computation for the same selector.
    """
    additions_by_class: dict[str, list[EntityNode]] = {}
    ordered_target_classes: list[str] = []

    if known_selectors is None:
        known_selectors = {}

    for item in selectors_value:
        selector_dict = None
        # Check if the item is a token (string) or a full selector definition (dict)
        if isinstance(item, str):
            selector_dict = known_selectors.get(item)
        elif isinstance(item, dict):
            selector_dict = item

        if selector_dict is None:
            continue

        target_class = selector_dict.get("target_class")
        pool = _get_entity_pool(entity_database, target_class)
        if not pool:
            continue
        cache_key = _get_selector_cache_key(selector_dict)
        additions = selector_cache.get(cache_key)
        if additions is None:
            additions = _apply_single_selector(pool, selector_dict)
            selector_cache[cache_key] = additions
        if target_class not in additions_by_class:
            additions_by_class[target_class] = []
            ordered_target_classes.append(target_class)
        additions_by_class[target_class].extend(additions)

    return additions_by_class, ordered_target_classes


def _merge_entities(
    existing: list[EntityNode],
    additions_by_class: dict[str, list[EntityNode]],
    ordered_target_classes: list[str],
    merge_mode: Literal["merge", "replace"],
) -> list[Any]:
    """Merge existing entities with selector additions based on merge mode."""
    base_entities: list[EntityNode] = []

    if merge_mode == "merge":  # explicit first, then selector additions
        base_entities.extend(existing)
        for target_class in ordered_target_classes:
            base_entities.extend(additions_by_class.get(target_class, []))

    else:  # replace: drop explicit items of targeted classes
        classes_to_update = set(ordered_target_classes)
        for item in existing:
            entity_type = _get_node_attribute(item, "private_attribute_entity_type_name")
            if entity_type not in classes_to_update:
                base_entities.append(item)
        for target_class in ordered_target_classes:
            base_entities.extend(additions_by_class.get(target_class, []))

    return base_entities


def _expand_node_selectors(
    entity_database: EntityDictDatabase,
    node: dict,
    selector_cache: dict,
    merge_mode: Literal["merge", "replace"],
    known_selectors: dict[str, dict] = None,
) -> None:
    """
    Expand selectors on one node and write results into stored_entities.

    - merge_mode="merge": keep explicit stored_entities first, then append selector results.
    - merge_mode="replace": replace explicit items of target classes affected by selectors.
    """
    selectors_value = node.get("selectors")
    if not (isinstance(selectors_value, list) and len(selectors_value) > 0):
        return

    additions_by_class, ordered_target_classes = _process_selectors(
        entity_database, selectors_value, selector_cache, known_selectors=known_selectors
    )

    existing = node.get("stored_entities", [])
    base_entities = _merge_entities(
        existing, additions_by_class, ordered_target_classes, merge_mode
    )

    node["stored_entities"] = base_entities

    # Replace string tokens with full selector definitions for pydantic validation
    if known_selectors:
        expanded_selectors = []
        for item in selectors_value:
            if isinstance(item, str):
                # ID string token
                if item in known_selectors:
                    expanded_selectors.append(known_selectors[item])
                else:
                    raise ValueError(
                        f"[Internal] Selector token '{item}' not found in known_selectors. "
                        "This may indicate a missing or invalid selector reference."
                    )
            else:
                expanded_selectors.append(item)
        node["selectors"] = expanded_selectors


def collect_and_tokenize_selectors_in_place(  # pylint: disable=too-many-branches
    params_as_dict: dict,
) -> dict:
    """
    Collect all matched/defined selectors into AssetCache and replace them with tokens (`selector_id`).

    This optimization reduces the size of the JSON and allows for efficient re-use of
    selector definitions.
    1. It traverses the `params_as_dict` to find all `EntitySelector` definitions (dicts with "name").
    2. It moves these definitions into `private_attribute_asset_cache["selectors"]`.
    3. It replaces the original dictionary definition in the `selectors` list with just the `selector_id` (token).
    """
    known_selectors = {}

    # Pre-populate from existing AssetCache if any
    asset_cache = params_as_dict.setdefault("private_attribute_asset_cache", {})
    if isinstance(asset_cache, dict):
        cached_selectors = asset_cache.get("used_selectors")
        if isinstance(cached_selectors, list):
            for s in cached_selectors:
                selector_id = s.get("selector_id")
                if selector_id is None:
                    selector_id = generate_uuid()
                    s["selector_id"] = selector_id
                known_selectors[selector_id] = s

    queue = deque([params_as_dict])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            selectors = node.get("selectors")
            new_selectors = []
            for item in selectors or ():
                selector_id = item.get("selector_id")
                known_selectors[selector_id] = item
                new_selectors.append(selector_id)
            if selectors is not None:
                node["selectors"] = new_selectors

            # Recurse
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)

        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    # Update AssetCache
    if isinstance(asset_cache, dict):
        asset_cache.pop("used_selectors", None)
        asset_cache["used_selectors"] = list(known_selectors.values())
    return params_as_dict


def expand_entity_selectors_in_place(
    entity_database: EntityDictDatabase,
    params_as_dict: dict,
    *,
    merge_mode: Literal["merge", "replace"] = "merge",
) -> dict:
    """Traverse params_as_dict and expand any EntitySelector in place.

    How caching works
    -----------------
    - Each selector must provide a unique name. We build a cross-tree
      cache key as ("name", target_class, name).
    - For every node that contains a non-empty `selectors` list, we compute the
      additions once per unique cache key, store the expanded list of entity
      dicts in `selector_cache`, and reuse it for subsequent nodes that reference
      the same selector name and target_class.
    - This avoids repeated pool scans and matcher compilation across the tree
      while preserving stable result ordering.

    Token Support
    -------------
    This function now also builds a `known_selectors` map from `private_attribute_asset_cache["selectors"]`.
    This map is passed down to `_process_selectors` to allow resolving string tokens back to their
    full selector definitions.

    Merge policy
    ------------
    - merge_mode="merge" (default): keep explicit `stored_entities` first, then
      append selector results; duplicates (if any) can be handled later by the
      materialization/dedup stage.
    - merge_mode="replace": for classes targeted by selectors in the node,
      drop explicit items of those classes and use selector results instead.
    """
    # Build known_selectors map from AssetCache if available
    asset_cache = params_as_dict.get("private_attribute_asset_cache")
    known_selectors = _collect_known_selectors_from_asset_cache(asset_cache)

    queue: deque[Any] = deque([params_as_dict])
    selector_cache: dict = {}
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            _expand_node_selectors(
                entity_database,
                node,
                selector_cache=selector_cache,
                merge_mode=merge_mode,
                known_selectors=known_selectors,
            )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    return params_as_dict
