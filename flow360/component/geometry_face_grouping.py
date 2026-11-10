"""
Client-side glue for rule-based face grouping.

The rule language and its evaluation are schema-owned and client-free
(``flow360_schema.geometry.selection`` — grammar — and
``selection_evaluation`` — the topology-tree walk). Rule-based
grouping is DRAFT STATE: authored via
``create_draft(face_grouping=FaceGroupingRules(...))``, resolved against the
geometry's topology tree, materialized into the draft's isolated entity_info
copy, and persisted automatically at submission. The ``Geometry`` cloud asset
stays immutable — its only grouping role is serving ``topology_tree``.
"""

from __future__ import annotations

from flow360_schema.geometry.selection_rules import resolve_group_assignments
from flow360_schema.models.entities.surface_entities import Surface

from flow360.exceptions import Flow360ValueError

# The Web UI stores custom/rule-based groups under this attribute tag.
CUSTOM_TAG = "rule-based-grouping"


def faces_by_group(assignment: dict) -> dict:
    """Invert a resolved ``{face_id: group_name}`` partition into
    ``{group_name: [face_id, ...]}`` — one entry per Surface the draft exposes.
    Group names are the Surface names as-is: the schema has already refused any
    group whose faces span more than one model file."""
    groups: dict = {}
    for face_id, group_name in assignment.items():
        groups.setdefault(group_name, []).append(face_id)
    return groups


def register_group_assignments(entity_info, assignment: dict) -> None:
    """Materialize a resolved ``{face_id: group_name}`` partition (from
    ``resolve_group_assignments``) into an entity_info as the ``rule-based-grouping``
    face attribute — one Surface per group. Intended for a DRAFT's isolated
    entity_info copy (the Geometry cloud asset stays immutable); activating the
    tag is the caller's normal grouping-selection path."""
    valid = set(entity_info.all_face_ids)
    unknown = [face_id for face_id in assignment if face_id not in valid]
    if unknown:
        raise Flow360ValueError(
            f"{len(unknown)} resolved face id(s) are not in the draft's entity_info "
            f"(e.g. {unknown[:3]}). Topology-tree / entity-info face-id mismatch — "
            "the tree uuids must equal the entity face ids."
        )
    surfaces = [
        Surface(
            name=group_name,
            private_attribute_id=group_name,
            private_attribute_tag_key=CUSTOM_TAG,
            private_attribute_sub_components=face_ids,
        )
        for group_name, face_ids in faces_by_group(assignment).items()
    ]
    if CUSTOM_TAG not in entity_info.face_attribute_names:
        entity_info.face_attribute_names.append(CUSTOM_TAG)
    # face_attribute_names/grouped_faces are independently-defaulted pydantic fields
    # (e.g. [] vs [[]]) and not guaranteed to be the same length on a fresh
    # entity_info, so pad before indexing rather than assuming positional alignment.
    while len(entity_info.grouped_faces) < len(entity_info.face_attribute_names):
        entity_info.grouped_faces.append([])
    entity_info.grouped_faces[entity_info.face_attribute_names.index(CUSTOM_TAG)] = surfaces


def apply_face_grouping_rules(rules, tree: dict, entity_info) -> str:
    """Resolve `rules` against `tree` and materialize the resulting groups into
    `entity_info`, returning the face-grouping tag the caller should activate.

    Takes the tree rather than the geometry it came from — the grouping mechanism
    has no business knowing about cloud resources.
    """
    register_group_assignments(entity_info, resolve_group_assignments(rules, tree))
    return CUSTOM_TAG
