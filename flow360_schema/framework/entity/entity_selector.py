"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

import logging
import re
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, cast

import pydantic as pd
from typing_extensions import Self

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_utils import (
    DEFAULT_NOT_MERGED_TYPES,
    compile_glob_cached,
    deduplicate_entities,
    generate_uuid,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_list import EntityList

# These corresponds to the private_attribute_entity_type_name of supported entity types.
TargetClass = Literal["Surface", "Edge", "GenericVolume", "GeometryBodyGroup"]

EntityNode = EntityBase | dict[str, Any]


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
    value: str | list[str] = pd.Field()
    # Applies only to matches/notMatches; default to glob if not specified explicitly.
    non_glob_syntax: Literal["regex"] | None = pd.Field(
        None,
        description="If specified, the pattern (`value`) will be treated "
        "as a non-glob pattern with the specified syntax.",
    )

    @pd.field_validator("value", mode="before")
    @classmethod
    def _normalize_membership_value(cls, value: Any, info: pd.ValidationInfo) -> Any:
        """Normalize single-string membership input to a one-item list."""
        if info.data.get("operator") in ("any_of", "not_any_of") and isinstance(value, str):
            return [value]
        return value


class EntitySelector(Flow360BaseModel):
    """Entity selector for an EntityList.

    - target_class chooses the entity pool
    - logic combines child predicates (AND = intersection, OR = union)
    """

    target_class: TargetClass = pd.Field()
    description: str | None = pd.Field(None, description="Customizable description of the selector.")
    selector_id: str = pd.Field(
        default_factory=generate_uuid,
        description="[Internal] Unique identifier for the selector.",
        frozen=True,
    )
    # Unique name for global reuse
    name: str = pd.Field(description="Unique name for this selector.")
    logic: Literal["AND", "OR"] = pd.Field("AND")
    children: list[Predicate] = pd.Field()

    @pd.validate_call
    def match(
        self,
        pattern: str,
        *,
        attribute: Literal["name"] = "name",
    ) -> Self:
        """Append a matches predicate (glob pattern) and return self for chaining."""
        self.children.append(
            Predicate(  # type: ignore[call-arg]
                attribute=attribute,
                operator="matches",
                value=pattern,
            )
        )
        return self

    @pd.validate_call
    def not_match(
        self,
        pattern: str,
        *,
        attribute: Literal["name"] = "name",
    ) -> Self:
        """Append a not-matches predicate (glob pattern) and return self for chaining."""
        self.children.append(
            Predicate(  # type: ignore[call-arg]
                attribute=attribute,
                operator="not_matches",
                value=pattern,
            )
        )
        return self

    @pd.validate_call
    def any_of(self, values: list[str], *, attribute: Literal["name"] = "name") -> Self:
        """Append an in predicate and return self for chaining."""
        self.children.append(Predicate(attribute=attribute, operator="any_of", value=values))  # type: ignore[call-arg]
        return self

    @pd.validate_call
    def not_any_of(self, values: list[str], *, attribute: Literal["name"] = "name") -> Self:
        """Append a notIn predicate and return self for chaining."""
        self.children.append(Predicate(attribute=attribute, operator="not_any_of", value=values))  # type: ignore[call-arg]
        return self


########## SELECTOR CLASSES ##########


class SurfaceSelector(EntitySelector):
    """
    Pattern-based selector for Surface entities. Stores matching rules,
    enabling reusable SimulationParams templates.

    Example
    -------
    >>> # Simple: match surfaces by glob pattern
    >>> fl.SurfaceSelector(name="wing_surfaces").match("wing*")
    >>> # With OR logic: match surfaces matching either pattern
    >>> sel = fl.SurfaceSelector(
    ...     name="wing_and_tail", logic="OR",
    ...     description="Wing and tail surfaces"
    ... ).match("wing*").match("tail*")

    ====
    """

    target_class: Literal["Surface"] = pd.Field("Surface", frozen=True)
    children: list[Predicate] = pd.Field(default_factory=list)


class EdgeSelector(EntitySelector):
    """
    Pattern-based selector for Edge entities. Stores matching rules,
    enabling reusable SimulationParams templates.

    Example
    -------
    >>> # Simple: match edges by glob pattern
    >>> fl.EdgeSelector(name="leading_edges").match("leading_edge*")
    >>> # With OR logic: match edges by name list
    >>> sel = fl.EdgeSelector(
    ...     name="selected_edges", logic="OR",
    ...     description="Edges for refinement"
    ... ).any_of(["edge1", "edge2"])

    ====
    """

    target_class: Literal["Edge"] = pd.Field("Edge", frozen=True)
    children: list[Predicate] = pd.Field(default_factory=list)


class VolumeSelector(EntitySelector):
    """
    Pattern-based selector for volume zone entities. Stores matching rules,
    enabling reusable SimulationParams templates.

    Example
    -------
    >>> # Simple: match volumes by glob pattern
    >>> fl.VolumeSelector(name="fluid_zones").match("fluid*")
    >>> # With OR logic: match specific zones by name
    >>> sel = fl.VolumeSelector(
    ...     name="porous_zones", logic="OR",
    ...     description="Zones for porous medium"
    ... ).any_of(["zone1", "zone2"])

    ====
    """

    target_class: Literal["GenericVolume"] = pd.Field("GenericVolume", frozen=True)
    children: list[Predicate] = pd.Field(default_factory=list)


class BodyGroupSelector(EntitySelector):
    """
    Pattern-based selector for body group entities. Stores matching rules,
    enabling reusable SimulationParams templates.

    Example
    -------
    >>> # Simple: match body groups by glob pattern
    >>> fl.BodyGroupSelector(name="rotor_bodies").match("rotor*")
    >>> # With OR logic: match specific bodies by name
    >>> sel = fl.BodyGroupSelector(
    ...     name="rotating_bodies", logic="OR",
    ...     description="Bodies for rotation"
    ... ).any_of(["body1", "body2"])

    ====
    """

    target_class: Literal["GeometryBodyGroup"] = pd.Field("GeometryBodyGroup", frozen=True)
    children: list[Predicate] = pd.Field(default_factory=list)


########## EXPANSION IMPLEMENTATION ##########


@lru_cache(maxsize=2048)
def _compile_regex_cached(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _get_node_attribute(entity: Any, attribute: str) -> Any:
    """Return attribute value from either dicts or entity objects."""
    if isinstance(entity, dict):
        return entity.get(attribute)
    return getattr(entity, attribute, None)


def _get_attribute_value(entity: Any, attribute: str) -> str | None:
    """Return the scalar string value of an attribute, or None if absent/unsupported.

    Only scalar string attributes are supported by this matcher layer for now.
    """
    val = _get_node_attribute(entity, attribute)
    if isinstance(val, str):
        return val
    return None


def _build_value_matcher(predicate: Predicate) -> Callable[[str | None], bool]:
    """
    Build a fast predicate(value: Optional[str])->bool matcher.

    Precompiles regex/glob and converts membership lists to sets for speed.
    """
    operator = predicate.operator
    value = predicate.value
    non_glob_syntax = predicate.non_glob_syntax

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
        values = {value} if isinstance(value, str) else set(value or [])

        def base_match(val: str | None) -> bool:
            return val in values

    elif base_operator == "matches":
        pattern = _compile_regex_cached(value) if non_glob_syntax == "regex" else compile_glob_cached(value)  # type: ignore[arg-type]

        def base_match(val: str | None) -> bool:
            return isinstance(val, str) and (pattern.fullmatch(val) is not None)

    else:

        def base_match(_val: str | None) -> bool:  # type: ignore[misc]
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
    ordered_children: list[Predicate],
) -> list[Any]:
    indices: set[int] = set()
    for predicate in ordered_children:
        attribute = predicate.attribute
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
    ordered_children: list[Predicate],
    indices_by_attribute: dict[str, dict[str, list[int]]],
) -> list[Any]:
    candidate_indices: set[int] | None = None

    def _matched_indices_for_predicate(predicate: Predicate, current_candidates: set[int] | None) -> set[int]:
        operator = predicate.operator
        attribute = predicate.attribute
        if operator == "any_of":
            idx_map = indices_by_attribute.get(attribute)
            if idx_map is not None:
                result: set[int] = set()
                for v in predicate.value or []:
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
        candidate_indices = matched if candidate_indices is None else candidate_indices.intersection(matched)
        if not candidate_indices:
            return []

    assert candidate_indices is not None
    if len(candidate_indices) * 4 < len(pool):
        return [pool[i] for i in sorted(candidate_indices)]
    return [pool[i] for i in range(len(pool)) if i in candidate_indices]


def _apply_single_selector(pool: list[EntityNode], selector: EntitySelector) -> list[EntityNode]:
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
    logic = selector.logic
    children = selector.children

    # Fast path: empty predicates -> return nothing. Empty children is actually misuse.
    if not children:
        return []

    # Predicate ordering (AND only): cheap/selective first
    def _cost(predicate: Predicate) -> int:
        op = predicate.operator
        order = {
            "any_of": 0,
            "matches": 1,
            "not_any_of": 2,
            "not_matches": 3,
        }
        return order.get(op, 10)

    ordered_children = children if logic == "OR" else sorted(children, key=_cost)

    # Optional per-attribute indices for in
    attributes_needing_index = {p.attribute for p in ordered_children if p.operator == "any_of"}
    indices_by_attribute: dict[str, dict[str, list[int]]] = (
        {attr: _build_index(pool, attr) for attr in attributes_needing_index} if attributes_needing_index else {}
    )

    if logic == "OR":
        # Favor a full scan for OR to preserve predictable union behavior
        # and avoid over-indexing that could complicate ordering.
        result = _apply_or_selector(pool, ordered_children)
    else:
        result = _apply_and_selector(pool, ordered_children, indices_by_attribute)

    if not result:
        name = selector.name
        target_class = selector.target_class
        logger.warning(
            "Entity selector '%s' (target_class=%s) matched 0 entities. "
            "Please check if the entity name or pattern is correct.",
            name,
            target_class,
        )

    return result


def _get_selector_cache_key(selector: EntitySelector) -> tuple[str, str]:
    """
    Return the cache key for a selector: requires unique name.

    We mandate a unique identifier per selector; use ("name", target_class, name)
    for stable global reuse. If neither `name` is provided, fall back to a
    structural key so different unnamed selectors won't collide.
    """
    # selector_id is always present and unique by schema.
    return ("selector_id", selector.selector_id)


def _process_selectors(
    registry: Any,
    selectors_list: list[Any],
    selector_cache: dict[str, Any],
    *,
    expansion_map: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[EntityNode]], list[str]]:
    """Process selectors and return additions grouped by class.

    This function iterates over a list of materialized selectors (EntitySelector-like objects
    or plain dicts) and applies them over the entity registry.
    Results are cached in `selector_cache` to avoid re-computation for the same selector.

    Parameters:
        registry: EntityRegistry instance containing entities.
        selectors_list: List of selector definitions (materialized; no string tokens).
        selector_cache: Cache for selector results.
        expansion_map: Type expansion mapping. If None, uses DEFAULT_TARGET_CLASS_EXPANSION_MAP
                       from the local schema module (lazy import).

    Returns:
        Tuple of (additions_by_class dict, ordered_target_classes list).
    """
    # Set default expansion map via lazy import from the local schema module.
    if expansion_map is None:
        from flow360_schema.framework.entity.entity_expansion_config import (
            DEFAULT_TARGET_CLASS_EXPANSION_MAP,
        )

        expansion_map = DEFAULT_TARGET_CLASS_EXPANSION_MAP

    additions_by_class: dict[str, list[EntityNode]] = {}
    ordered_target_classes: list[str] = []

    for item in selectors_list:
        if not isinstance(item, EntitySelector):
            raise TypeError(f"[Internal] selectors_list must contain EntitySelector objects. Got: {type(item)}")
        selector: EntitySelector = item
        target_class = selector.target_class

        # Use expansion map to get multiple type names
        expanded_type_names = expansion_map.get(target_class, [target_class])
        entities = registry.find_by_type_name(expanded_type_names)
        if not entities:
            continue
        cache_key = _get_selector_cache_key(selector)
        additions = selector_cache.get(cache_key)  # type: ignore[call-overload]
        if additions is None:
            additions = _apply_single_selector(entities, selector)
            selector_cache[cache_key] = additions  # type: ignore[index]
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
    not_merged_types: frozenset[str] = DEFAULT_NOT_MERGED_TYPES,
) -> list[Any]:
    """Merge existing entities with selector additions based on merge mode.

    Type filtering is intentionally out of scope here. Callers are expected to
    filter selector additions before passing them into this merge helper.
    """
    candidates: list[EntityNode] = []

    if merge_mode == "merge":  # explicit first, then selector additions
        candidates.extend(existing)
        for target_class in ordered_target_classes:
            candidates.extend(additions_by_class.get(target_class, []))

    else:  # replace: drop explicit items of targeted classes
        classes_to_update = set(ordered_target_classes)
        for item in existing:
            entity_type = _get_node_attribute(item, "private_attribute_entity_type_name")
            if entity_type not in classes_to_update:
                candidates.append(item)
        for target_class in ordered_target_classes:
            candidates.extend(additions_by_class.get(target_class, []))

    # Deduplication logic (same as materialize_entities_and_selectors_in_place)
    return deduplicate_entities(
        candidates,
        not_merged_types=not_merged_types,
    )


def expand_entity_list_selectors(
    registry: Any,
    entity_list: Any,
    *,
    selector_cache: dict[str, Any] | None = None,
    merge_mode: Literal["merge", "replace"] = "merge",
    expansion_map: dict[str, list[str]] | None = None,
) -> list[EntityNode]:
    """
    Expand selectors in a single EntityList within an EntityRegistry context.

    Parameters
    ----------
    expansion_map : Optional type expansion mapping for selectors.

    Notes
    -----
    - This function does NOT modify the input EntityList.
    - selector_cache can be shared across multiple calls to reuse selector results.
    - This helper performs raw selector expansion only. It does not apply any
      EntityList-specific type filtering.
    """
    stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
    raw_selectors = list(getattr(entity_list, "selectors", []) or [])

    if selector_cache is None:
        selector_cache = {}

    if not raw_selectors:
        return stored_entities

    additions_by_class, ordered_target_classes = _process_selectors(
        registry,
        raw_selectors,
        selector_cache,
        expansion_map=expansion_map,
    )
    return _merge_entities(
        stored_entities,
        additions_by_class,
        ordered_target_classes,
        merge_mode,
    )


def resolve_entity_list_selectors(
    registry: Any,
    entity_list: "EntityList",
    *,
    selector_cache: dict[str, Any] | None = None,
    merge_mode: Literal["merge", "replace"] = "merge",
    expansion_map: dict[str, list[str]] | None = None,
) -> list[EntityBase]:
    """
    Expand selectors for a concrete EntityList and filter selector additions to its valid types.

    Explicit ``stored_entities`` are assumed to have already been pre-filtered by
    EntityList's ``before`` validator. Only selector additions are type-filtered here
    before they are merged into the resolved entity list.
    """
    from flow360_schema.framework.entity.entity_list import (
        _filter_entities_by_valid_types,
        _get_valid_entity_type_info,
    )

    stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
    raw_selectors = list(getattr(entity_list, "selectors", []) or [])

    if selector_cache is None:
        selector_cache = {}

    if not raw_selectors:
        return stored_entities

    additions_by_class, ordered_target_classes = _process_selectors(
        registry,
        raw_selectors,
        selector_cache,
        expansion_map=expansion_map,
    )

    raw_selector_entities = []
    filtered_additions_by_class: dict[str, list[EntityNode]] = {}
    for target_class in ordered_target_classes:
        additions = additions_by_class.get(target_class, [])
        entity_additions = [item for item in additions if isinstance(item, EntityBase)]
        raw_selector_entities.extend(entity_additions)
        filtered_additions_by_class[target_class] = cast(
            list[EntityNode],
            _filter_entities_by_valid_types(
                entity_list.__class__,
                entity_additions,
            ),
        )

    merged_entities = _merge_entities(
        stored_entities,
        filtered_additions_by_class,
        ordered_target_classes,
        merge_mode,
    )
    if raw_selector_entities and not merged_entities:
        valid_types, _ = _get_valid_entity_type_info(entity_list.__class__)
        valid_type_name_list = [valid_type.__name__ for valid_type in valid_types]
        raise ValueError(f"Can not find any valid entity of type {valid_type_name_list} from the input.")
    return merged_entities


def collect_and_tokenize_selectors_in_place(
    params_as_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Collect selector definitions from a fresh JSON params dict and replace them with tokens (`selector_id`).

    Intended usage:
    - Input is the result of `model_dump(mode="json")` before selector tokenization.
    - This is a single-pass preprocessing step for upload/validation payload preparation.
    - It is not intended to be re-run on a params dict whose `selectors` entries are already tokens.

    Behavior:
    1. Traverse `params_as_dict` to find inline `EntitySelector` definitions.
    2. Move these definitions into `private_attribute_asset_cache["used_selectors"]`.
    3. Replace each selector definition in a `selectors` list with its `selector_id` token.
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

    queue: deque[dict[str, Any] | list[Any]] = deque([params_as_dict])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            selectors = node.get("selectors")
            new_selectors: list[Any] = []
            for item in selectors or ():
                selector_id = item.get("selector_id")
                if selector_id is None:
                    selector_id = generate_uuid()
                    item["selector_id"] = selector_id
                known_selectors[selector_id] = item
                new_selectors.append(selector_id)
            if selectors is not None:
                node["selectors"] = new_selectors

            # Recurse
            for value in node.values():
                if value is asset_cache:
                    continue
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
