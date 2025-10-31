"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Literal, Optional, Union, get_args

import pydantic as pd
from typing_extensions import Self

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
    # Unique name for global reuse (aka tag)
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

    This is intended to strip off differences between root resources and
    ensure the expansion has a uniform data interface.
    """

    surfaces: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    generic_volumes: list[dict] = field(default_factory=list)
    geometry_body_groups: list[dict] = field(default_factory=list)


########## API IMPLEMENTATION ##########


def _validate_selector_factory_common(
    method_name: str,
    *,
    name: str,
    attribute: str,
    logic: str,
    syntax: Optional[str] = None,
) -> None:
    """Validate common arguments for SelectorFactory methods.

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
            selector_name=name, entity_class=cls, logic=logic
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
            selector_name=name, entity_class=cls, logic=logic
        )
        selector.not_match(pattern, attribute=attribute, syntax=syntax)
        return selector

    @classmethod
    def any_of(
        cls,
        values: List[str],
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        logic: Literal["AND", "OR"] = "AND",
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
        _validate_selector_values("any_of", values)  # type: ignore[arg-type]

        selector = generate_entity_selector_from_class(
            selector_name=name, entity_class=cls, logic=logic
        )
        selector.any_of(values, attribute=attribute)
        return selector

    @classmethod
    def not_any_of(
        cls,
        values: List[str],
        /,
        *,
        name: str,
        attribute: Literal["name"] = "name",
        logic: Literal["AND", "OR"] = "AND",
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
            selector_name=name, entity_class=cls, logic=logic
        )
        selector.not_any_of(values, attribute=attribute)
        return selector


def generate_entity_selector_from_class(
    selector_name: str, entity_class: type, logic: Literal["AND", "OR"] = "AND"
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

    return EntitySelector(name=selector_name, target_class=class_name, logic=logic, children=[])


########## EXPANSION IMPLEMENTATION ##########
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
    """Compile an extended-glob pattern via wcmatch to a fullmatch-ready regex.

    We enable extended glob features including brace expansion, extglob groups,
    and globstar. We intentionally avoid PATHNAME semantics because entity
    names are not paths in this context, and we keep case-sensitive matching to
    remain predictable across platforms.
    """
    # Strong requirement: wcmatch must be present to support full glob features.
    try:
        # pylint: disable=import-outside-toplevel
        from wcmatch import fnmatch as wfnmatch
    except Exception as exc:  # pragma: no cover - explicit failure path
        raise RuntimeError(
            "wcmatch is required for extended glob support. Please install 'wcmatch>=10.0'."
        ) from exc

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


def _get_attribute_value(entity: dict, attribute: str) -> Optional[str]:
    """Return the scalar string value of an attribute, or None if absent/unsupported.

    Only scalar string attributes are supported by this matcher layer for now.
    """
    val = entity.get(attribute)
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
            pattern = _compile_glob_cached(value)

        def base_match(val: Optional[str]) -> bool:
            return isinstance(val, str) and (pattern.fullmatch(val) is not None)

    else:

        def base_match(_val: Optional[str]) -> bool:
            return False

    if negate:
        return lambda val: not base_match(val)
    return base_match


def _build_index(pool: list[dict], attribute: str) -> dict[str, list[int]]:
    """Build an index for in lookups on a given attribute."""
    value_to_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(pool):
        val = item.get(attribute)
        if isinstance(val, str):
            value_to_indices.setdefault(val, []).append(idx)
    return value_to_indices


def _apply_or_selector(
    pool: list[dict],
    ordered_children: list[dict],
) -> list[dict]:
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
    pool: list[dict],
    ordered_children: list[dict],
    indices_by_attribute: dict[str, dict[str, list[int]]],
) -> list[dict]:
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
        return _apply_or_selector(pool, ordered_children)

    return _apply_and_selector(pool, ordered_children, indices_by_attribute)


def _get_selector_cache_key(selector_dict: dict) -> tuple:
    """
    Return the cache key for a selector: requires unique name/tag.

    We mandate a unique identifier per selector; use ("name", target_class, name)
    for stable global reuse. If neither `name` nor `tag` is provided, fall back to a
    structural key so different unnamed selectors won't collide.
    """
    tclass = selector_dict.get("target_class")
    name = selector_dict.get("name")
    if name:
        return ("name", tclass, name)

    logic = selector_dict.get("logic", "AND")
    children = selector_dict.get("children") or []

    def _normalize_value(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    preds = tuple(
        (
            p.get("attribute", "name"),
            p.get("operator"),
            _normalize_value(p.get("value")),
            p.get("non_glob_syntax"),
        )
        for p in children
        if isinstance(p, dict)
    )
    return ("struct", tclass, logic, preds)


def _process_selectors(
    entity_database: EntityDictDatabase,
    selectors_value: list,
    selector_cache: dict,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Process selectors and return additions grouped by class."""
    additions_by_class: dict[str, list[dict]] = {}
    ordered_target_classes: list[str] = []

    for selector_dict in selectors_value:
        if not isinstance(selector_dict, dict):
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
    existing: list[dict],
    additions_by_class: dict[str, list[dict]],
    ordered_target_classes: list[str],
    merge_mode: Literal["merge", "replace"],
) -> list[dict]:
    """Merge existing entities with selector additions based on merge mode."""
    base_entities: list[dict] = []

    if merge_mode == "merge":  # explicit first, then selector additions
        base_entities.extend(existing)
        for target_class in ordered_target_classes:
            base_entities.extend(additions_by_class.get(target_class, []))

    else:  # replace: drop explicit items of targeted classes
        classes_to_update = set(ordered_target_classes)
        for item in existing:
            entity_type = item.get("private_attribute_entity_type_name")
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
        entity_database, selectors_value, selector_cache
    )

    existing = node.get("stored_entities", [])
    base_entities = _merge_entities(
        existing, additions_by_class, ordered_target_classes, merge_mode
    )

    node["stored_entities"] = base_entities
    # node["selectors"] = selectors_value
    print(">>> selectors: ", node["selectors"])


def expand_entity_selectors_in_place(
    entity_database: EntityDictDatabase,
    params_as_dict: dict,
    *,
    merge_mode: Literal["merge", "replace"] = "merge",
) -> dict:
    """Traverse params_as_dict and expand any EntitySelector in place.

    How caching works
    -----------------
    - Each selector must provide a unique name (or tag). We build a cross-tree
      cache key as ("name", target_class, name).
    - For every node that contains a non-empty `selectors` list, we compute the
      additions once per unique cache key, store the expanded list of entity
      dicts in `selector_cache`, and reuse it for subsequent nodes that reference
      the same selector name and target_class.
    - This avoids repeated pool scans and matcher compilation across the tree
      while preserving stable result ordering.

    Merge policy
    ------------
    - merge_mode="merge" (default): keep explicit `stored_entities` first, then
      append selector results; duplicates (if any) can be handled later by the
      materialization/dedup stage.
    - merge_mode="replace": for classes targeted by selectors in the node,
      drop explicit items of those classes and use selector results instead.
    """
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
            )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    return params_as_dict
