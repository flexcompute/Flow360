"""
Rule evaluation over topology trees — the data walk of the geometry-selection
grammar (see docs/geometry_selection_grammar.md; the program walk — parse /
validate / serialize — lives in selection.py).

Service-importable by design: everything here operates on plain topology-tree
dicts (the ``GET /v2/geometries/<id>/topology-tree`` payload) with no Flow360
client dependency. A web service resolves a rule with only this package:

    from flow360_schema.geometry.selection_evaluation import resolve_rule

    face_ids = resolve_rule('root.bodies(name="wing*").faces()', topology_tree)

Concepts (semantics-parity with the Web UI's utils.ts, pinned by the resolution
corpus in tests/geometry/face_grouping/):

  * rule DESCRIPTIONS (``SelectionChain``/``CombinedSelection``/the ``root``
    token) live in geometry/selection.py and carry NO geometry state. This module
    is the other half of that split: pure evaluation functions taking the
    description and the topology tree as explicit arguments —
    ``resolve_selection(selection, tree)`` / ``resolve_rule(rule_string, tree)``.
    Kernel-legality checks happen here, at resolution, when a tree first exists.
  * a RULE is a CHAIN of navigation steps (``SelectionChain``), each narrowing the
    tree by a method + step condition (keyword match fields or a predicate)::

        root.bodies(name="wing*").faces(color="#00FF00")
        #    ^ container (keep chaining)   ^ leaf shortcut (terminal)

    A chain may stop at a container (``root.bodies(name="wing*")``): the stored rule
    stays as written, and a face group built from it descends to every groupable
    leaf under those containers (Face on CAD, Solid on ASCII STL, Boundary on
    UGRID/CGNS; a binary STL's childless ModelFile is its own leaf) — the
    kernel-agnostic "everything under here". ``resolve_selection`` without
    ``leaf_faces`` returns the containers themselves.
  * a GROUP is the UNION (OR) of rules (``rule_a | rule_b``, a ``CombinedSelection``);
    difference ``-`` subtracts faces.
"""

from __future__ import annotations

import logging
from typing import Any

from flow360_schema.geometry.selection import (
    CombinedSelection,
    NodePredicate,
    SelectionChain,
    color_to_hex,
    rule_to_selection,
)

logger = logging.getLogger(__name__)

# ---- ported constants (utils.ts) -------------------------------------------
GROUPABLE_LEAF_TYPES = {"Face", "Solid", "Boundary"}
# All B-rep body subtypes `.bodies()` matches. The frontend lists the typed
# subtypes; we also accept the generic "Body" some emitters still produce.
BODY_TYPES = [
    "SolidBody",
    "SheetBody",
    "SurfaceBody",
    "ShellBody",
    "WireBody",
    "SingleWireBody",
    "Body",
]
LEAF_SHORTCUT_TYPE_MAP = {"faces": "Face", "solids": "Solid", "boundaries": "Boundary"}
CONTAINER_SHORTCUT_TYPE_MAP = {
    "assemblies": ["Assembly"],
    "parts": ["Part"],
    "bodies": BODY_TYPES,
    "zones": ["Zone"],
}

# Which type-specific navigation shortcuts each ModelFile ``modelingKernel`` has a level
# for, derived from that kernel's topology-tree hierarchy. children/descendants/model_files are
# structure-agnostic and always allowed; a shortcut not listed for the geometry's
# kernel is rejected (e.g. assemblies()/parts() on an EGADSGeometry, faces() on a UGRID).
_GENERIC_METHODS = {"children", "descendants", "model_files"}
_MODELING_KERNEL_METHODS = {
    "CADGeometry": {"assemblies", "parts", "bodies", "faces"},
    "EGADSGeometry": {"bodies", "faces"},
    "BinarySTL": set(),  # the ModelFile itself is the lone anonymous groupable leaf
    "ASCIISTL": {"solids"},
    "UGRID": {"boundaries"},
    "CGNS": {"zones", "boundaries"},
}
# Human label used in the "<label> has no <method>" error for an invalid type shortcut.
_MODELING_KERNEL_LABEL = {
    "CADGeometry": "CAD",
    "EGADSGeometry": "EGADS",
    "BinarySTL": "Binary STL model",
    "ASCIISTL": "ASCII STL model",
    "UGRID": "UGRID model",
    "CGNS": "CGNS model",
}
# Kernels whose ModelFile is the sole groupable leaf (no navigable hierarchy at all):
# children()/descendants() also raise, as "<label> has no <method> nodes".
_NO_CHILD_KERNEL_LABEL = {
    "BinarySTL": "Binary STL file",
}


# ---- tree walking helpers ----------------------------------------------------
def _selection_id(tree_node: dict[str, Any]) -> str | None:
    """The id a resolved selection carries for this node.

    Transitional contract: ``attributes.legacyId`` (the compatibility id the
    manifest/mesher/entity-info speak) when present, else ``uuid`` — the
    fallback keeps trees converted before the identity triple working
    unchanged (their uuid IS the legacy id). At the native cutover consumers
    switch to the (resource, modelIndex, uuid) triple and this helper loses
    the legacyId branch.
    """
    legacy = (tree_node.get("attributes") or {}).get("legacyId")
    return legacy or tree_node.get("uuid") or None


def _is_groupable_leaf(tree_node: dict[str, Any]) -> bool:
    if not _selection_id(tree_node):
        return False
    if tree_node.get("type") in GROUPABLE_LEAF_TYPES:
        return True
    return tree_node.get("type") == "ModelFile" and not tree_node.get("children")


def _collect_by_type(tree_node: dict[str, Any], type_: str, out: list[dict[str, Any]]) -> None:
    if tree_node.get("type") == type_:
        out.append(tree_node)
    for child in tree_node.get("children") or []:
        _collect_by_type(child, type_, out)


def _collect_by_types(tree_node: dict[str, Any], types: set[str], out: list[dict[str, Any]]) -> None:
    if tree_node.get("type") in types:
        out.append(tree_node)
    for child in tree_node.get("children") or []:
        _collect_by_types(child, types, out)


def _collect_leaf_uuids(tree_node: dict[str, Any], out: list[str]) -> None:
    selection_id = _selection_id(tree_node)
    if selection_id and _is_groupable_leaf(tree_node):
        out.append(selection_id)
    for child in tree_node.get("children") or []:
        _collect_leaf_uuids(child, out)


def all_leaf_ids(tree: dict[str, Any]) -> list[str]:
    """Every groupable-leaf uuid in the tree (Face/Solid/Boundary/childless
    ModelFile), ordered, de-duplicated — the universe a face partition covers."""
    out: list[str] = []
    _collect_leaf_uuids(tree, out)
    return dedup(out)


def leaf_ids_by_model_file(tree: dict[str, Any]) -> dict[str, list[str]]:
    """``{model file name: its groupable-leaf uuids}`` for every ModelFile in the tree
    (a childless ModelFile, e.g. a binary STL, is its own leaf). Keyed by name because
    rules address a model file by name and the mesher groups bodies by it; the tree
    contract makes that key safe — a ModelFile's name is its uploaded file name, unique
    within a geometry, and every leaf sits under one."""
    files: list[dict[str, Any]] = []
    _collect_by_type(tree, "ModelFile", files)
    return {node["name"]: all_leaf_ids(node) for node in files}


def _step_matches(tree_node: dict[str, Any], predicate: NodePredicate | None) -> bool:
    """Step condition test: the predicate tree (grammar profile) or pass-all."""
    return predicate is None or predicate.matches(tree_node)


# ---- evaluator (semantics-parity with utils.ts, pinned by the resolution corpus) ----
def evaluate_selection(expr: dict[str, Any], tree: dict[str, Any], leaf_faces: bool = False) -> list[str]:
    """Walk the TOPOLOGY TREE (the geometry data) step-by-step -> ordered,
    de-duplicated uuids (one rule).

    Distinct from the expression-AST walk: the engine walks the parsed *program*
    once (building the SelectionChain via the proxy methods); this function then walks
    the *data* — applying each navigation step and its predicate to tree nodes.

    leaf_faces decides what an UNTERMINATED chain (no .faces()/.solids()/.boundaries()
    step) returns: True descends the current scope to its groupable leaf FACES (what
    a face group needs); False (default, for `.ids`) returns the CURRENT-SCOPE
    nodes' own uuids, so root.assemblies()/parts()/bodies() yield the container uuids.
    """
    root_type = tree.get("type")
    if root_type == "GeometryResource":
        current = [c for c in (tree.get("children") or []) if c.get("type") == "ModelFile"]
    elif root_type == "ModelFile":
        current = [tree]
    else:
        current = [tree]
    # Kernel-legality is checked against the model files IN SCOPE, and a model_files()
    # step narrows that scope: root.model_files(name="*.stl").solids() must be judged
    # by the STL models only, exactly like typed navigation is walked on them only.
    scope_models = _models_under(current)

    for step in expr["steps"]:
        method = step.method
        predicate = step.predicate
        # kernel-legality is an evaluation-time concern: descriptions are
        # tree-agnostic, so the check runs here, scoped like the walk itself
        _require_method_supported_for(scope_models, method)
        if method == "children":
            result: list[dict[str, Any]] = []
            for n in current:
                result.extend(n.get("children") or [])
            current = [n for n in result if _step_matches(n, predicate)]
        elif method == "descendants":
            result = []
            for n in current:
                for child in n.get("children") or []:
                    result.append(child)
                    stack = list(child.get("children") or [])
                    while stack:
                        x = stack.pop()
                        result.append(x)
                        stack.extend(x.get("children") or [])
            current = [n for n in result if _step_matches(n, predicate)]
        elif method == "model_files":
            files: list[dict[str, Any]] = []
            for n in current:
                _collect_by_type(n, "ModelFile", files)
            current = [n for n in files if _step_matches(n, predicate)]
            scope_models = _models_under(current)
            if files and not current:
                # An empty file match silences the whole chain and is almost always
                # a typo -- but it must stay a WARNING: an instance-agnostic rule
                # applied to a tree without matching files legitimately resolves to
                # an empty set (e.g. one arm of a cross-kernel union).
                available = ", ".join(sorted(n.get("name") or "?" for n in files))
                logger.warning(
                    "A model_files() step matched no model file; this tree has: %s. "
                    "The chain resolves to an empty selection.",
                    available,
                )
        elif method in LEAF_SHORTCUT_TYPE_MAP:  # faces / solids / boundaries -> terminal
            target = LEAF_SHORTCUT_TYPE_MAP[method]
            leaves: list[dict[str, Any]] = []
            for n in current:
                _collect_by_type(n, target, leaves)
            out = [
                selection_id
                for leaf in leaves
                if (selection_id := _selection_id(leaf)) and _step_matches(leaf, predicate)
            ]
            return list(dict.fromkeys(out))  # dedupe, keep first-seen order
        elif method in CONTAINER_SHORTCUT_TYPE_MAP:  # assemblies / parts / bodies / zones
            targets = set(CONTAINER_SHORTCUT_TYPE_MAP[method])
            collected: list[dict[str, Any]] = []
            for n in current:
                _collect_by_types(n, targets, collected)
            current = [n for n in collected if _step_matches(n, predicate)]

    # End of chain, no terminal leaf shortcut. Face mode descends the current scope to
    # its groupable leaves (a childless ModelFile, e.g. a binary STL, is its own leaf);
    # node mode (default, for `.ids`) returns the current-scope nodes' own uuids
    # (container uuids for assemblies/parts/bodies).
    if leaf_faces:
        leaf_out: list[str] = []
        for n in current:
            _collect_leaf_uuids(n, leaf_out)
        return dedup(leaf_out)
    return dedup(selection_id for n in current if (selection_id := _selection_id(n)))


# ---- modeling-kernel-driven structural restrictions ------------------------
def _kernel_label(kernel: str) -> str:
    """Human label for a modeling kernel in error messages (e.g. 'BinarySTL' -> 'Binary
    STL model', 'EGADSGeometry' -> 'EGADS')."""
    if kernel in _MODELING_KERNEL_LABEL:
        return _MODELING_KERNEL_LABEL[kernel]
    return kernel[: -len("Geometry")] if kernel.endswith("Geometry") else kernel


def _models_under(nodes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """[(model_file_name, modeling_kernel), ...] for every kernel-carrying ModelFile
    within the given scope nodes (the nodes themselves included)."""
    out: list[tuple[str, str]] = []
    for node in nodes:
        found: list[dict[str, Any]] = []
        _collect_by_type(node, "ModelFile", found)
        out.extend((n.get("name") or "", n["modelingKernel"]) for n in found if n.get("modelingKernel"))
    return out


def models_with_kernels(tree: dict[str, Any], model_files: tuple[str, ...] | list[str] = ()) -> list[tuple[str, str]]:
    """[(model_file_name, modeling_kernel), ...] for each ModelFile in the topology tree
    carrying a modelingKernel, optionally restricted to the ModelFile(s) named in
    ``model_files`` (matched on the tree node's name). Empty model_files means every model
    in the resource."""
    wanted = set(model_files or ())
    out: list[tuple[str, str]] = []

    def walk(tree_node: Any) -> None:
        if isinstance(tree_node, dict):
            if (
                tree_node.get("type") == "ModelFile"
                and tree_node.get("modelingKernel")
                and (not wanted or tree_node.get("name") in wanted)
            ):
                out.append((tree_node.get("name") or "", tree_node["modelingKernel"]))
            for child in tree_node.get("children") or []:
                walk(child)

    walk(tree)
    return out


def _kernel_disallows(kernel: str, method: str) -> bool:
    """Whether a model of ``kernel`` lacks a level for ``method``. Generic shortcuts are
    allowed except children()/descendants() on a kernel whose ModelFile is the sole leaf
    (BinarySTL)."""
    if method in ("children", "descendants"):
        return kernel in _NO_CHILD_KERNEL_LABEL
    if method in _GENERIC_METHODS:
        return False
    allowed = _MODELING_KERNEL_METHODS.get(kernel)
    return allowed is not None and method not in allowed


def _single_model_message(kernel: str, method: str) -> str:
    if method in ("children", "descendants") and kernel in _NO_CHILD_KERNEL_LABEL:
        return f"{_NO_CHILD_KERNEL_LABEL[kernel]} has no {method} nodes"
    return f"{_kernel_label(kernel)} has no {method}"


def require_method_supported(tree: dict[str, Any], method: str, model_files: tuple[str, ...] | list[str] = ()) -> None:
    """Reject a navigation shortcut the in-scope model kernel(s) have no level for.
    ``model_files`` scopes the check to specific ModelFile(s) (a
    ``root.model_files(...)`` scope); empty means the whole resource. With a single model in scope the error
    names that model's kernel ("<label> has no <method>"); across multiple models a
    whole-resource shortcut is rejected as soon as ANY model lacks the level. Unknown/
    absent kernels stay permissive so pre-kernel trees keep working."""
    _require_method_supported_for(models_with_kernels(tree, model_files), method)


def _require_method_supported_for(models_found: list[tuple[str, str]], method: str) -> None:
    offending = [(name, kernel) for name, kernel in models_found if _kernel_disallows(kernel, method)]
    if not offending:
        return
    if len(models_found) > 1:
        raise ValueError(f"Failed to call .{method} since some models do not have {method}")
    _name, kernel = offending[0]
    raise ValueError(_single_model_message(kernel, method))


# ---- topology-tree rendering -------------------------------------------------
def _node_label(tree_node: dict[str, Any], show_attributes: bool) -> str:
    """`Type` followed by each field the node actually carries, as `key=value`:
    name, color, material, attributes (when non-empty)."""
    parts = [tree_node.get("type") or "?"]
    if tree_node.get("name"):
        parts.append(f'name="{tree_node["name"]}"')
    if tree_node.get("colorRGB"):
        parts.append(f'color="{color_to_hex(tree_node["colorRGB"])}"')
    material = tree_node.get("material") or {}
    if isinstance(material, dict) and material.get("name"):
        parts.append(f'material="{material["name"]}"')
    if show_attributes and tree_node.get("attributes"):
        body = ", ".join(f"{k}={v}" for k, v in tree_node["attributes"].items())
        parts.append(f"attributes={{{body}}}")
    return "  ".join(parts)


def _render_tree(root: dict[str, Any], show_attributes: bool, max_depth: int | None) -> str:
    lines = [_node_label(root, show_attributes)]

    def walk(tree_node: dict[str, Any], prefix: str, depth: int) -> None:
        children = tree_node.get("children") or []
        if max_depth is not None and depth > max_depth:
            if children:
                lines.append(f"{prefix}└─ … ({len(children)} more)")
            return
        for i, child in enumerate(children):
            last = i == len(children) - 1
            lines.append(prefix + ("└─ " if last else "├─ ") + _node_label(child, show_attributes))
            walk(child, prefix + ("   " if last else "│  "), depth + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def _color_summary(root: dict[str, Any]) -> str:
    leaves: list[dict[str, Any]] = []
    _collect_by_type(root, "Face", leaves)
    counts: dict[str, int] = {}
    for face in leaves:
        key = color_to_hex(face.get("colorRGB")) or "(none)"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return ", ".join(f"{c}×{n}" for c, n in sorted(counts.items()))


def render_topology_tree(tree: dict[str, Any], show_attributes: bool = True, max_depth: int | None = None) -> str:
    """Printable ASCII rendering of a topology tree (the same tree the rule engine
    walks), with a trailing summary of the distinct face colors present."""
    out = _render_tree(tree, show_attributes, max_depth)
    colors = _color_summary(tree)
    if colors:
        out += f"\n\nface colors: {colors}"
    return out


# ---- evaluation entry points ---------------------------------------------------
def dedup(ids: Any) -> list[Any]:
    """Order-preserving de-duplication (first occurrence wins)."""
    return list(dict.fromkeys(ids))


def resolve_selection(selection: Any, tree: dict[str, Any], *, leaf_faces: bool = False) -> list[str]:
    """Evaluate a rule DESCRIPTION against a topology tree -> ordered,
    de-duplicated uuids. THE fluent-object entry point (the string twin is
    ``resolve_rule``); descriptions are tree-agnostic, so the tree is always an
    explicit argument here — there is no bound-selection object.

    leaf_faces decides what an UNTERMINATED chain (no .faces()/.solids()/
    .boundaries() step) yields: True descends to groupable leaf FACES (what a
    face group needs); False (default) yields the current-scope nodes' own
    uuids, so root.assemblies()/parts()/bodies() give the container uuids.
    Union semantics: first-seen dedup; difference preserves the left's order."""
    if isinstance(selection, CombinedSelection):
        left = resolve_selection(selection.left, tree, leaf_faces=leaf_faces)
        right = resolve_selection(selection.right, tree, leaf_faces=leaf_faces)
        if selection.op == "BitOr":
            return dedup(left + right)
        drop = set(right)
        return [f for f in left if f not in drop]
    if isinstance(selection, SelectionChain):
        return evaluate_selection(selection.expr, tree, leaf_faces=leaf_faces)
    raise TypeError(f"Cannot resolve {type(selection).__name__} — expected a rule description")


def evaluate_rule_on_tree(rule_string: str, topology_tree: dict[str, Any]) -> list[str]:
    """Parse + validate + evaluate a rule string against a topology tree ->
    ordered uuids (faces for a terminal chain, container uuids otherwise —
    the resolution-corpus semantics)."""
    return resolve_selection(rule_to_selection(rule_string), topology_tree)


def resolve_rule(rule_string: str, topology_tree: dict[str, Any]) -> list[str]:
    """Service entry point: rule string + topology tree -> ordered, de-duplicated
    LEAF FACE ids (container selections auto-descend, matching face-group semantics)."""
    return resolve_selection(rule_to_selection(rule_string), topology_tree, leaf_faces=True)


# ---- tree inspection helpers ----------------------------------------------------
def modeling_kernel(tree: dict[str, Any]) -> Any:
    """The tree's ModelFile ``modelingKernel`` -- one of CADGeometry, EGADSGeometry,
    BinarySTL, ASCIISTL, UGRID, CGNS. A single string for a single-file tree, a set
    when a multi-file tree mixes kernels, or None when the tree predates the
    modelingKernel field."""
    kernels = {kernel for _name, kernel in models_with_kernels(tree)}
    if not kernels:
        return None
    return next(iter(kernels)) if len(kernels) == 1 else kernels


def models(tree: dict[str, Any]) -> dict[str, SelectionChain]:
    """The resource's model files as ``{model_file_name: description}``, keyed by
    each ModelFile node's ``name`` (companion files that are not ModelFile nodes,
    e.g. a UGRID's .mapbc, are excluded). Each value is an (unbound) SelectionChain
    scoped to that single model — resolve it with ``resolve_selection``."""
    out: dict[str, SelectionChain] = {}

    def walk(tree_node: Any) -> None:
        if isinstance(tree_node, dict):
            if tree_node.get("type") == "ModelFile" and tree_node.get("name"):
                out[tree_node["name"]] = SelectionChain().model_files(name=tree_node["name"])
            for child in tree_node.get("children") or []:
                walk(child)

    walk(tree)
    return out
