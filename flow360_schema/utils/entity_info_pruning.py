"""Raw-dict pruning of non-active entity_info groupings for pipeline consumers.

``GeometryEntityInfo`` stores one complete partition of every face/edge/body per
attribute name, but validation and translation consume almost exclusively the
grouping selected by each ``*_group_tag``. Pipeline entrypoints call
:func:`prune_entity_info_for_pipeline` on the raw ``simulation.json`` dict before
``validate_model`` so pydantic never parses the groupings nothing will read.

The prune must only ever run on a transient, run-scoped dict — never on a dict
that is written back to a persisted ``simulation.json``, and never in the
geometry conversion pipeline (the entity_info producer). Client-side flows
(regrouping, forking, ``apply_simulation_setting_to_entity_info``, merging)
need every grouping and must not prune.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The GAI geometry-download fallback and legacy mesher file-path resolution read
# the "groupByFile" body grouping by literal name (`_get_processed_file_list`),
# regardless of the active body_group_tag.
_BODY_GROUPINGS_ALWAYS_KEPT = ("groupByFile",)

# The mirror validator's face-to-body fallback
# (`get_body_group_to_surface_mapping`) reads the "groupByBodyId" face grouping
# by literal name when `bodies_face_edge_ids` is empty or absent.
_MIRROR_FALLBACK_FACE_GROUPING = "groupByBodyId"

_EDGE_KEYS = ("edge_ids", "edge_attribute_names", "grouped_edges")


def _contains_edge_references(params_as_dict: dict[str, Any], entity_info: dict[str, Any]) -> bool:
    """Whether anything outside ``entity_info`` references Edge entities.

    Counts compact reference groups ({"type": "Edge", "ids": [...]}), entity
    selectors ({"target_class": "Edge", ...}), and inline entity dicts from
    pre-compact-ref JSON versions (the updater rewrites those against the
    active edge grouping, so it must survive).
    """
    stack: list[Any] = [params_as_dict]
    while stack:
        node = stack.pop()
        if node is entity_info:
            continue
        if isinstance(node, dict):
            if "Edge" in (
                node.get("type"),
                node.get("target_class"),
                node.get("private_attribute_entity_type_name"),
            ):
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _prune_grouping_kind(
    entity_info: dict[str, Any], tag_key: str, names_key: str, grouped_key: str, extra_keep: tuple[str, ...]
) -> int:
    """Shrink one kind's ``*_attribute_names``/``grouped_*`` pair in lockstep.

    Returns the number of groupings dropped. No-op (returns 0) when the group
    tag is missing or unknown — prune only what is provably safe.
    """
    group_tag = entity_info.get(tag_key)
    names = entity_info.get(names_key) or []
    if not group_tag or group_tag not in names:
        return 0
    keep = {group_tag, *extra_keep}
    keep_indices = [index for index, name in enumerate(names) if name in keep]
    if len(keep_indices) == len(names):
        return 0
    grouped = entity_info[grouped_key]
    entity_info[names_key] = [names[index] for index in keep_indices]
    entity_info[grouped_key] = [grouped[index] for index in keep_indices]
    return len(names) - len(keep_indices)


def prune_entity_info_for_pipeline(params_as_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop entity_info groupings no pipeline consumer will read, in place.

    Keeps, per kind: the ``*_group_tag``-selected grouping; the "groupByFile"
    body grouping; the "groupByBodyId" face grouping when the mirror validator
    can fall back to it; the active edge grouping only when the params
    reference Edge entities, otherwise every edge field. A kind whose tag is
    absent or not listed in its attribute names is left untouched.
    """
    asset_cache = params_as_dict.get("private_attribute_asset_cache") or {}
    entity_info = asset_cache.get("project_entity_info") or {}
    if entity_info.get("type_name") != "GeometryEntityInfo":
        return params_as_dict

    mirror_status = asset_cache.get("mirror_status") or {}
    face_keep = (
        (_MIRROR_FALLBACK_FACE_GROUPING,)
        if not entity_info.get("bodies_face_edge_ids") and mirror_status.get("mirrored_geometry_body_groups")
        else ()
    )
    dropped = {
        "face": _prune_grouping_kind(entity_info, "face_group_tag", "face_attribute_names", "grouped_faces", face_keep),
        "body": _prune_grouping_kind(
            entity_info, "body_group_tag", "body_attribute_names", "grouped_bodies", _BODY_GROUPINGS_ALWAYS_KEPT
        ),
    }
    if _contains_edge_references(params_as_dict, entity_info):
        dropped["edge"] = _prune_grouping_kind(
            entity_info, "edge_group_tag", "edge_attribute_names", "grouped_edges", ()
        )
    else:
        dropped["edge"] = sum(key in entity_info for key in _EDGE_KEYS)
        for key in _EDGE_KEYS:
            entity_info.pop(key, None)

    if any(dropped.values()):
        logger.info(
            "Pruned entity_info groupings before validation: %s",
            ", ".join(f"{kind}={count}" for kind, count in dropped.items()),
        )
    return params_as_dict
