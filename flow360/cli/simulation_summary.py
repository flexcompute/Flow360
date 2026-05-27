"""Generic SimulationParams compaction for CLI inspection.

The summary command intentionally validates and normalizes through SimulationParams so it can
remove SDK defaults and show user intent. Raw SimulationParams JSON remains available through
`simulation-params get`.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import OrderedDict

_PRIVATE_PREFIX = "private_attribute_"
_ENTITY_COLLECTION_KEYS = ("stored_entities", "selectors")
_GROUP_LABEL_KEYS = {"name"}
_SAMPLE_LIMIT = 10
_DEFAULT_REL_TOL = 1e-12
_DEFAULT_ABS_TOL = 1e-15


def summarize_simulation(simulation_params: dict) -> dict:
    """Validate SimulationParams JSON and return a compact JSON projection."""

    display_dict, normalized_dict, default_dict = _load_summary_dicts(simulation_params)
    compact_display = _compact_value(display_dict)
    if default_dict is None:
        return compact_display
    return _prune_defaults(
        compact_display,
        _compact_value(normalized_dict),
        _compact_value(default_dict),
    )


def _load_summary_dicts(simulation_params: dict) -> tuple[dict, dict, dict | None]:
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.simulation_params import SimulationParams

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    clear_variable_space = False
    try:
        params_dict = SimulationParams._sanitize_params_dict(  # pylint: disable=protected-access
            copy.deepcopy(simulation_params)
        )
        params_dict, _ = SimulationParams._update_param_dict(  # pylint: disable=protected-access
            params_dict
        )
        root_item_type = _infer_root_item_type(params_dict)
        unit_system_name = _unit_system_name(params_dict)
        length_unit = _project_length_unit(params_dict)
        clear_variable_space = _initialize_summary_variable_space(params_dict)
        params_dict = _strip_private_cache(params_dict)
        display_dict = _strip_private_cache(copy.deepcopy(simulation_params))
        params = SimulationParams(file_content=copy.deepcopy(params_dict))
        normalized_dict = _strip_private_cache(params.model_dump(mode="json", exclude_none=True))
        default_dict = _default_params_dict(unit_system_name, length_unit, root_item_type)
        return display_dict, normalized_dict, default_dict
    finally:
        if clear_variable_space:
            _clear_summary_variable_space()
        logging.disable(previous_disable_level)


def _initialize_summary_variable_space(params_dict):
    # pylint: disable=import-outside-toplevel,broad-exception-caught
    from flow360.component.simulation.services import initialize_variable_space

    variable_context = params_dict.get("private_attribute_asset_cache", {}).get("variable_context")
    if not variable_context:
        return False
    try:
        initialize_variable_space(params_dict, use_clear_context=True)
    except Exception:
        _clear_summary_variable_space()
        # Summary only needs cached names to resolve while SimulationParams is rebuilt.
        _initialize_summary_variable_placeholders(variable_context)
    return True


def _initialize_summary_variable_placeholders(variable_context):
    # pylint: disable=import-outside-toplevel,broad-exception-caught
    from flow360_schema.framework.expression.registry import default_context

    for variable_info in variable_context:
        if not isinstance(variable_info, dict):
            continue
        name = variable_info.get("name")
        if not name:
            continue
        try:
            default_context.set_value(
                name,
                _summary_variable_value(variable_info.get("value")),
            )
        except Exception:
            continue

        description = variable_info.get("description")
        if description is not None:
            default_context.set_metadata(name, "description", description)

        metadata = variable_info.get("metadata")
        if metadata is not None:
            default_context.set_metadata(name, "metadata", metadata)


def _summary_variable_value(value):
    # pylint: disable=import-outside-toplevel,broad-exception-caught
    from flow360_schema.framework.expression import Expression

    if isinstance(value, dict) and "expression" in value:
        expression = value["expression"]
        output_units = value.get("output_units")
        try:
            return Expression(expression=expression, output_units=output_units)
        except Exception:
            return Expression.model_construct(
                expression=expression,
                output_units=output_units,
            )
    return value


def _clear_summary_variable_space():
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.services import clear_context

    clear_context()


def _infer_root_item_type(params_dict):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.services import (
        _parse_root_item_type_from_simulation_json,
    )

    try:
        return _parse_root_item_type_from_simulation_json(param_as_dict=params_dict)
    except ValueError:
        return "VolumeMesh" if params_dict.get("meshing") is None else "Geometry"


def _unit_system_name(params_dict):
    unit_system = params_dict.get("unit_system")
    if isinstance(unit_system, dict):
        return unit_system.get("name") or "SI"
    return unit_system or "SI"


def _project_length_unit(params_dict):
    project_length_unit = params_dict.get("private_attribute_asset_cache", {}).get(
        "project_length_unit"
    )
    if isinstance(project_length_unit, dict):
        return project_length_unit.get("units") or "m"
    return "m"


def _default_params_dict(unit_system_name, length_unit, root_item_type):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.services import get_default_params

    try:
        return _strip_private_cache(
            get_default_params(unit_system_name, length_unit, root_item_type)
        )
    except (RuntimeError, ValueError, TypeError):
        return None


def _compact_value(value):
    if isinstance(value, dict):
        if _is_entity_collection(value):
            return _compact_entity_collection(value)
        if set(value) == {"items"}:
            return _compact_value(value["items"])
        return _compact_mapping(value)

    if isinstance(value, list):
        return _compact_sequence(value)

    return value


def _compact_mapping(value):
    compacted = OrderedDict()
    for key, child in value.items():
        if _should_drop_key(key):
            continue
        compacted[key] = _compact_value(child)
    return _clean_empty(compacted)


def _compact_sequence(value):
    compacted = [_compact_value(item) for item in value]
    compacted = [item for item in compacted if item not in ({}, [], None)]

    if not compacted:
        return []

    if all(isinstance(item, dict) for item in compacted):
        return _group_compacted_mappings(compacted)

    if len(compacted) > _SAMPLE_LIMIT:
        return {"_count": len(compacted), "_sample": compacted[:_SAMPLE_LIMIT]}

    return compacted


def _group_compacted_mappings(items):
    groups = OrderedDict()
    for item in items:
        signature = _group_signature(item)
        bucket = groups.setdefault(
            signature,
            {
                "count": 0,
                "representative": item,
                "labels": [],
                "entity_summaries": OrderedDict(),
            },
        )
        bucket["count"] += 1
        bucket["labels"].extend(_extract_labels(item))
        _collect_entity_summaries(item, bucket["entity_summaries"])

    if len(groups) == len(items) and len(items) <= _SAMPLE_LIMIT:
        return items

    grouped_items = [_serialize_group(bucket) for bucket in groups.values()]
    if len(grouped_items) > _SAMPLE_LIMIT:
        return {"_count": len(grouped_items), "_sample": grouped_items[:_SAMPLE_LIMIT]}
    return grouped_items


def _serialize_group(bucket):
    if bucket["count"] == 1:
        return bucket["representative"]

    representative = copy.deepcopy(bucket["representative"])
    _drop_group_variable_fields(representative)
    representative["_count"] = bucket["count"]

    labels = _unique(bucket["labels"])
    if labels:
        representative["_names"] = _sample(labels)

    for path, names in bucket["entity_summaries"].items():
        _set_path(representative, path, _entity_summary(names))

    return _clean_empty(representative)


def _compact_entity_collection(value):
    names = []
    for key in _ENTITY_COLLECTION_KEYS:
        for entity in value.get(key) or []:
            names.append(_entity_label(entity))
    return _entity_summary(names)


def _entity_summary(names):
    unique_names = _unique([name for name in names if name])
    if not unique_names:
        return {"_count": 0}
    return {"_count": len(unique_names), "_sample": _sample(unique_names)}


def _entity_label(entity):
    if isinstance(entity, dict):
        return (
            entity.get("name") or entity.get("id") or entity.get("type") or entity.get("type_name")
        )
    return str(entity)


def _group_signature(value):
    signature_value = _strip_group_variable_fields(value)
    return json.dumps(signature_value, sort_keys=True, separators=(",", ":"))


def _strip_group_variable_fields(value):
    if isinstance(value, dict):
        return {
            key: _strip_group_variable_fields(child)
            for key, child in value.items()
            if key not in _GROUP_LABEL_KEYS and not _is_entity_summary(child)
        }
    if isinstance(value, list):
        return [_strip_group_variable_fields(item) for item in value]
    return value


def _drop_group_variable_fields(value):
    if not isinstance(value, dict):
        return
    for key in list(value):
        if key in _GROUP_LABEL_KEYS or _is_entity_summary(value[key]):
            value.pop(key)
            continue
        _drop_group_variable_fields(value[key])


def _extract_labels(value):
    labels = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _GROUP_LABEL_KEYS and child:
                labels.append(child)
            elif isinstance(child, (dict, list)):
                labels.extend(_extract_labels(child))
    elif isinstance(value, list):
        for child in value:
            labels.extend(_extract_labels(child))
    return labels


def _collect_entity_summaries(value, summaries, path=()):
    if isinstance(value, dict):
        if _is_entity_summary(value):
            summaries.setdefault(path, []).extend(value.get("_sample") or [])
            return
        for key, child in value.items():
            _collect_entity_summaries(child, summaries, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_entity_summaries(child, summaries, (*path, index))


def _set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        if isinstance(target, dict):
            target = target.setdefault(key, OrderedDict())
        elif isinstance(target, list) and isinstance(key, int) and key < len(target):
            target = target[key]
        else:
            return
    if not path:
        return
    final_key = path[-1]
    if isinstance(target, dict):
        target[final_key] = replacement
    elif isinstance(target, list) and isinstance(final_key, int) and final_key < len(target):
        target[final_key] = replacement


def _prune_defaults(  # pylint: disable=too-many-return-statements
    display_value,
    normalized_value,
    default_value,
    *,
    keep_type_marker=False,
    depth=0,
):
    if default_value is None:
        return display_value
    if _matches_default(normalized_value, default_value):
        if keep_type_marker:
            return _type_marker(display_value)
        return None

    # Wire-format dicts are atomic: pruning `units` while keeping `value`
    # would produce an unloadable payload because the active wire format
    # requires both keys.
    if isinstance(display_value, dict) and set(display_value.keys()) == {"value", "units"}:
        return display_value

    if (
        isinstance(display_value, dict)
        and isinstance(normalized_value, dict)
        and isinstance(default_value, dict)
    ):
        pruned = OrderedDict()
        for key, child in display_value.items():
            child_keep_type_marker = _is_type_marker_key(key) or (
                depth == 0 and bool(_type_marker(child))
            )
            if key in default_value:
                child = _prune_defaults(
                    child,
                    normalized_value.get(key),
                    default_value.get(key),
                    keep_type_marker=child_keep_type_marker,
                    depth=depth + 1,
                )
            elif _is_absent_default_like(child):
                child = None
            if child not in ({}, [], None):
                pruned[key] = child

        marker = _type_marker(display_value)
        if (
            marker
            and (pruned or keep_type_marker)
            and not any(_is_type_marker_key(key) for key in pruned)
        ):
            pruned = OrderedDict([*marker.items(), *pruned.items()])
        return _clean_empty(pruned)

    if (
        isinstance(display_value, list)
        and isinstance(normalized_value, list)
        and isinstance(default_value, list)
    ):
        return _prune_default_sequence(display_value, normalized_value, default_value, depth=depth)

    return display_value


def _prune_default_sequence(display_items, normalized_items, default_items, *, depth):
    matched_default_indices = set()
    pruned_items = []
    for index, display_item in enumerate(display_items):
        normalized_item = normalized_items[index] if index < len(normalized_items) else None
        default_index = _find_default_match(normalized_item, default_items, matched_default_indices)
        if default_index is None:
            pruned_items.append(display_item)
            continue
        matched_default_indices.add(default_index)
        pruned_item = _prune_defaults(
            display_item,
            normalized_item,
            default_items[default_index],
            keep_type_marker=True,
            depth=depth + 1,
        )
        if pruned_item not in ({}, [], None):
            pruned_items.append(pruned_item)
    return pruned_items


def _find_default_match(normalized_item, default_items, matched_indices):
    normalized_marker = _type_marker(normalized_item)
    normalized_name = normalized_item.get("name") if isinstance(normalized_item, dict) else None
    for index, default_item in enumerate(default_items):
        if index in matched_indices:
            continue
        if _matches_default(normalized_item, default_item):
            return index
        if not normalized_marker or normalized_marker != _type_marker(default_item):
            continue
        default_name = default_item.get("name") if isinstance(default_item, dict) else None
        if normalized_name is None or default_name is None or normalized_name == default_name:
            return index
    return None


def _type_marker(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in value if _is_type_marker_key(key)}


def _is_type_marker_key(key):
    return key in {"type", "type_name", "output_type", "refinement_type"}


def _matches_default(value, default):
    if _is_number(value) and _is_number(default):
        return math.isclose(
            float(value),
            float(default),
            rel_tol=_DEFAULT_REL_TOL,
            abs_tol=_DEFAULT_ABS_TOL,
        )
    if isinstance(value, dict) and isinstance(default, dict):
        if value.keys() != default.keys():
            return False
        return all(_matches_default(value[key], default[key]) for key in value)
    if isinstance(value, list) and isinstance(default, list):
        if len(value) != len(default):
            return False
        return all(_matches_default(child, default[index]) for index, child in enumerate(value))
    return value == default


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_absent_default_like(value):
    return value in (None, False, 0, 0.0, [], {})


def _is_entity_collection(value):
    if not any(key in value for key in _ENTITY_COLLECTION_KEYS):
        return False
    entities = []
    for key in _ENTITY_COLLECTION_KEYS:
        entities.extend(value.get(key) or [])
    return all(isinstance(entity, dict) for entity in entities)


def _is_entity_summary(value):
    return isinstance(value, dict) and set(value) <= {"_count", "_sample"} and "_count" in value


def _should_drop_key(key):
    return isinstance(key, str) and (key.startswith(_PRIVATE_PREFIX) or key == "_id")


def _strip_private_cache(value):
    if isinstance(value, dict):
        return {
            key: _strip_private_cache(child)
            for key, child in value.items()
            if key not in {"private_attribute_asset_cache", "private_attribute_dict"}
        }
    if isinstance(value, list):
        return [_strip_private_cache(item) for item in value]
    return value


def _sample(items):
    return items[:_SAMPLE_LIMIT]


def _unique(items):
    seen = set()
    unique = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _clean_empty(value):
    if isinstance(value, dict):
        cleaned = OrderedDict()
        for key, child in value.items():
            cleaned_child = _clean_empty(child)
            if cleaned_child in ({}, [], None):
                continue
            cleaned[key] = cleaned_child
        return dict(cleaned)
    if isinstance(value, list):
        cleaned = []
        for item in value:
            cleaned_item = _clean_empty(item)
            if cleaned_item in ({}, [], None):
                continue
            cleaned.append(cleaned_item)
        return cleaned
    return value
