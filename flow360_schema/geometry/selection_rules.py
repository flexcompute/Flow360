"""Face-grouping rules payload (v3.0) and group-set resolution.

Author a group's rule in Python — ``root.faces(color="#00FF00")``,
``root.model_files(name="wing.step").faces()`` — with ``root`` and ``node`` from
``flow360_schema.geometry.selection`` (re-exported here, so one import serves:
``from flow360_schema.geometry.selection_rules import FaceGroupRule, FaceGroupingRules, root``;
the Flow360 client binds this module as ``fl.face_grouping``). The payload stores every rule as its
canonical rule text (see docs/geometry_selection_grammar.md): that text is the
wire format the Web UI reads and writes, and it is itself valid Python, so a
stored rule pastes straight back into client code. ``RuleText`` is the
geometry-side analogue of the math-expression ``Expression`` entry point: it
accepts a selection or its text, validates through ``parse_rule`` and stores
the canonical text.

Group-set resolution (``resolve_group_assignments``) is the partition layer
ABOVE the per-rule DSL: ordered first-match-wins over the group list, the
catch-all group (``is_remaining``) for the faces no rule matches — required as
soon as there is one, never a made-up name — an explicit warning, never
silence, when groups overlap, and two hard constraints: every rule group ends
up with at least one face, and every group, the catch-all included, stays
inside one model file (the surface mesher's body group) — or resolution raises.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic as pd

from flow360_schema.geometry.selection import RuleSelection, node, parse_rule, root, rule_to_string
from flow360_schema.geometry.selection_evaluation import all_leaf_ids, leaf_ids_by_model_file, resolve_rule

__all__ = ["FaceGroupRule", "FaceGroupingRules", "node", "resolve_group_assignments", "root"]

logger = logging.getLogger(__name__)
RULES_PAYLOAD_VERSION: Literal["3.0"] = "3.0"


def _serialize_fluent_rule(value: Any) -> Any:
    """Accept a fluent selection (typed, autocompleted authoring) by serializing it
    to its canonical rule text; strings pass through to grammar validation."""
    if isinstance(value, RuleSelection):
        return value.to_rule_string()
    return value


def _canonicalize_rule_text(value: str) -> str:
    """Validate (parse + grammar check) and normalize to the canonical form."""
    return rule_to_string(parse_rule(value))


# The Expression-analogue entry point: a fluent selection (``root.faces(...)``) or
# its rule text, validated by the grammar and stored as the canonical text
# (byte-stable across round trips). The stored type is ``str`` on purpose: the
# JSON schema and the Web UI see plain text.
RuleText = Annotated[
    str,
    pd.BeforeValidator(_serialize_fluent_rule),
    pd.AfterValidator(_canonicalize_rule_text),
]


class FaceGroupRule(pd.BaseModel):
    """One named group: a selection, or the catch-all marker.

    Write the rule as a selection::

        FaceGroupRule(name="green", rule=root.faces(color="#00FF00"))

    It is stored as canonical rule text (``rule`` reads back as ``str``); the text
    is accepted too, since it is what the wire carries.

    The catch-all (``is_remaining``) carries NO rule and no face snapshot — its
    membership is "everything not matched by a higher-priority group", re-derived
    at every resolution (a frozen list would lock it to one geometry and break
    reuse across revisions)."""

    model_config = pd.ConfigDict(populate_by_name=True)

    name: str = pd.Field(min_length=1)
    rule: RuleText | None = None
    is_remaining: bool = pd.Field(False, alias="isRemaining")

    if TYPE_CHECKING:
        # The authoring signature, for type checkers and IDE hover: a selection or its
        # text goes in, canonical text is what the field holds. Runtime init is pydantic's.
        def __init__(
            self, *, name: str, rule: RuleSelection | str | None = None, is_remaining: bool = False
        ) -> None: ...

    @pd.model_validator(mode="after")
    def _rule_xor_remaining(self) -> FaceGroupRule:
        if self.is_remaining == (self.rule is not None):
            raise ValueError(
                f"Group '{self.name}': provide exactly one of rule=root.faces(...) or "
                "is_remaining=True (the catch-all carries no rule)"
            )
        return self

    def __repr__(self) -> str:
        # Pasteable client Python: the stored rule text IS the fluent form, unquoted.
        body = "is_remaining=True" if self.is_remaining else f"rule={self.rule}"
        return f"FaceGroupRule(name={json.dumps(self.name)}, {body})"

    __str__ = __repr__


class FaceGroupingRules(pd.BaseModel):
    """The v3.0 rules payload: an ordered group list (order IS the priority)."""

    version: Literal["3.0"] = RULES_PAYLOAD_VERSION
    groups: list[FaceGroupRule]

    @pd.model_validator(mode="after")
    def _validate_group_list(self) -> FaceGroupingRules:
        names = [group.name for group in self.groups]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate group name(s): {sorted(duplicates)}")
        remaining = [i for i, group in enumerate(self.groups) if group.is_remaining]
        if len(remaining) > 1:
            raise ValueError("At most one catch-all (is_remaining) group is allowed")
        if remaining and remaining[0] != len(self.groups) - 1:
            raise ValueError(
                "The catch-all (is_remaining) group must be last — the group order is "
                "the match priority and the catch-all takes whatever is left"
            )
        return self

    def __repr__(self) -> str:
        return f"FaceGroupingRules(groups=[{', '.join(map(repr, self.groups))}])"

    __str__ = __repr__


def _reject_groups_left_without_faces(
    rules: FaceGroupingRules, assignment: dict[str, str], overlaps: dict[tuple[str, str], list[str]]
) -> None:
    """A rule group that ends up with no faces is a Surface the draft silently lacks: its
    rule matches nothing, or earlier groups claimed every face it matches. The catch-all
    is exempt — an empty remainder means the rules covered every face. Names every such
    group at once, each with its reason."""
    claimed = set(assignment.values())
    lines: list[str] = []
    for group in rules.groups:
        if group.is_remaining or group.name in claimed:
            continue
        shadowed_by = {winner: ids for (winner, loser), ids in overlaps.items() if loser == group.name}
        if not shadowed_by:
            lines.append(f"  - '{group.name}': its rule matches no face on this geometry")
            continue
        claims = ", ".join(
            f"'{winner}' ({len(ids)} face(s), e.g. {', '.join(ids[:3])})" for winner, ids in shadowed_by.items()
        )
        lines.append(f"  - '{group.name}': every face its rule matches went to an earlier group, {claims}")
    if not lines:
        return
    raise ValueError(
        "Face group(s) left with no faces:\n"
        + "\n".join(lines)
        + "\nFirst match wins, by group order: list a group before the groups that take its faces, or drop it."
    )


def _reject_groups_spanning_model_files(assignment: dict[str, str], topology_tree: dict[str, Any]) -> None:
    """The surface mesher refuses one patch shared across body groups (error 4082), and
    bodies are grouped by model file — so every face group, the catch-all included,
    must stay inside one model file. Names every offending group at once."""
    owner = {face_id: file for file, ids in leaf_ids_by_model_file(topology_tree).items() for face_id in ids}
    faces_by_group_and_file: dict[str, dict[str, list[str]]] = {}
    for face_id, group in assignment.items():  # insertion order = group order, catch-all last
        faces_by_group_and_file.setdefault(group, {}).setdefault(owner[face_id], []).append(face_id)
    spanning = {group: files for group, files in faces_by_group_and_file.items() if len(files) > 1}
    if not spanning:
        return
    lines = "\n".join(
        f"  - '{group}': "
        + ", ".join(f"{file} ({len(ids)} face(s), e.g. {', '.join(ids[:3])})" for file, ids in files.items())
        for group, files in spanning.items()
    )
    first_spanning_files = list(next(iter(spanning.values())))[:2]
    hint = " and ".join(f'root.model_files(name="{file}").faces(...)' for file in first_spanning_files)
    raise ValueError(
        "A face group spanning more than one model file is not supported:\n"
        f"{lines}\n"
        f"Write one group per model file and scope each rule to its file, e.g. {hint}. "
        "Faces left to the catch-all must all live in one file too."
    )


def resolve_group_assignments(rules: FaceGroupingRules, topology_tree: dict[str, Any]) -> dict[str, str]:
    """Resolve the ordered group list into the face partition: ``{face_id: group_name}``.

    First matching group wins; faces matched by no rule go to the catch-all group
    (``is_remaining``), and a payload without one is rejected as soon as a face is
    left over (``ValueError``) — nothing is grouped under a made-up name. Overlaps
    between groups are ALLOWED but never silent: one logged warning summarizes
    every (winning, losing) group pair with face counts and example ids. A rule
    group left with no faces — its rule matches nothing, or
    earlier groups claimed every face it matches — is rejected (``ValueError`` naming
    each such group and why; the catch-all may end up empty). A group whose faces land
    in more than one model file is rejected (``ValueError`` listing every such group,
    catch-all included) — the surface mesher cannot mesh one patch across body groups.
    """
    assignment: dict[str, str] = {}
    overlaps: dict[tuple[str, str], list[str]] = {}

    for group in rules.groups:
        if group.is_remaining:
            continue
        assert group.rule is not None  # guaranteed by the rule-xor-remaining validator
        for face_id in resolve_rule(group.rule, topology_tree):
            winner = assignment.get(face_id)
            if winner is None:
                assignment[face_id] = group.name
            elif winner != group.name:
                overlaps.setdefault((winner, group.name), []).append(face_id)

    if overlaps:
        details = "; ".join(
            f"'{winner}' over '{loser}': {len(face_ids)} face(s), e.g. {face_ids[:3]}"
            for (winner, loser), face_ids in overlaps.items()
        )
        logger.warning("Face-group rules overlap (first match wins, by group order): %s", details)

    unmatched = [face_id for face_id in all_leaf_ids(topology_tree) if face_id not in assignment]
    if unmatched:
        catch_all = next((group.name for group in rules.groups if group.is_remaining), None)
        if catch_all is None:
            raise ValueError(
                f"{len(unmatched)} face(s) match no rule, e.g. {', '.join(unmatched[:3])}. Add a catch-all "
                "group, FaceGroupRule(name=..., is_remaining=True), to collect them, or extend the rules."
            )
        assignment.update(dict.fromkeys(unmatched, catch_all))
    _reject_groups_left_without_faces(rules, assignment, overlaps)
    _reject_groups_spanning_model_files(assignment, topology_tree)
    return assignment
