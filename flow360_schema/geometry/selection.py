"""Geometry-selection grammar profile (see docs/geometry_selection_grammar.md).

The face-grouping rule language is a frozen subset of Python expression syntax.
This module is the single source of truth for that subset on the Python side:

- the grammar constants (navigation methods, match fields),
- the matching semantics (the schema's shared glob dialect, color normalization),
- the ``node`` field proxy and predicate objects — full-boolean step conditions
  (``faces((node.name == "*wing*") | ~(node.material == "steel"))``) built
  identically by live Python (operator overloading) and by engine evaluation of
  a parsed rule string,
- ``match_fields_to_predicate`` — the DETACHABLE kwargs sugar: keyword match
  fields desugar into the same predicate objects at the entry boundary, so the
  predicate tree is the only internal condition representation,
- ``validate_rule_ast`` — reject any parsed AST outside the subset, with a
  specific, actionable error (this is also the security boundary),
- ``parse_rule`` — parse + validate a rule string in one call,
- ``rule_to_string`` — the canonical serializer (double quotes, ``", "``
  separators, minimal parentheses); byte-stable for identical rules.

Evaluation of a full rule is NOT this module's job: callers bind ``root``
(and ``node``) in a fresh, local ``EvaluationContext`` and evaluate the
validated AST themselves. Predicates, however, carry their own node-level
evaluation (``NodePredicate.matches(tree_node)``).

A rule is an INSTANCE-AGNOSTIC description — a navigation path plus attribute
filters over the geometry hierarchy tree — and stays tree-agnostic until it is
evaluated against a particular geometry (via a draft, or ``resolve_selection``
/ ``resolve_rule`` in ``selection_evaluation``).
The grammar's root name is ``root`` (the root of the hierarchy tree);
``geometry`` is deliberately NOT used — that word names the cloud-resource
concept, which a rule is independent of.
"""

import colorsys
import json
import re
from dataclasses import dataclass
from typing import Any, cast

from flow360_schema.framework.expression.engine.ast_nodes import (
    BinOpNode,
    CallOnResultNode,
    ConstantNode,
    DictNode,
    ExpressionNode,
    ExpressionNodeType,
    ListNode,
    NameNode,
    StaticCallNode,
    SubscriptNode,
    UnaryOpNode,
)
from flow360_schema.framework.expression.engine.eval_context import EvaluationContext
from flow360_schema.framework.expression.engine.operators import BINARY_OPERATORS
from flow360_schema.framework.expression.engine.resolver import CallableResolver
from flow360_schema.framework.glob_pattern import compile_glob

ROOT_NAME = "root"
NODE_NAME = "node"
CONTAINER_METHODS = ("children", "descendants", "model_files", "assemblies", "parts", "bodies", "zones")
LEAF_METHODS = ("faces", "solids", "boundaries")
NAV_METHODS = CONTAINER_METHODS + LEAF_METHODS
MATCH_FIELDS = ("name", "type", "uuid", "color", "material", "attributes")

# One wording for both routes a user can reach '+' by (fluent __add__, rule string).
_SELECTIONS_ARE_SETS = "Selections are sets: use '|' (union) and '-' (difference), like Python sets; there is no '+'."
_EMPTY_RULE_MESSAGE = f"A rule is an expression starting from '{ROOT_NAME}', e.g. {ROOT_NAME}.faces()"

# Python precedence (loose -> tight) restricted to the grammar's operators:
# comparisons < `|` < `&` < arithmetic `-`; `~` is the tightest (unary).
_PRECEDENCE = {"Eq": 1, "NotEq": 1, "BitOr": 2, "BitAnd": 3, "Sub": 4}
_RULE_OPS = ("BitOr", "Sub")  # composition of selections
_PREDICATE_BOOL_OPS = ("BitAnd", "BitOr")
_COMPARE_OPS = ("Eq", "NotEq")
_INVERT_PRECEDENCE = 5


class RuleSyntaxError(ValueError):
    """A rule expression is outside the geometry-selection grammar."""


class _NoFilter:
    """'No filter given for this match field'. ``None`` cannot serve as the default:
    a field's value may legitimately be ``None`` (a condition in its own right --
    ``color=None`` filters on having no color), which has to stay distinguishable
    from not filtering at all."""

    def __repr__(self) -> str:
        return "<no filter>"


# Typed Any so a step's signature can read `name: str = _NOFILTER` -- the annotation
# states what a caller should pass, which is what IDE hover shows.
_NOFILTER: Any = _NoFilter()


def _given(**fields: Any) -> dict[str, Any]:
    """Just the match fields the caller actually passed."""
    return {key: value for key, value in fields.items() if value is not _NOFILTER}


# ---- matching semantics (shared by kwargs sugar and predicates) ---------------
def _compile_glob_or_raise(field: str, pattern: str) -> re.Pattern[str]:
    """The schema's one glob dialect (``framework/glob_pattern``: stdlib ``fnmatchcase``,
    case-sensitive, whole string), with its rejection of foreign spellings re-raised as
    the rule language's one error type."""
    try:
        return compile_glob(pattern)
    except ValueError as error:
        raise RuleSyntaxError(f"{field}: {error}") from error


def color_to_string(color: Any) -> str:
    """A node's colorRGB -> 'r,g,b' (ints). Tree colors are 0-255 ints; round
    defensively so float jitter still matches."""
    if not color:
        return ""
    return ",".join(str(int(round(float(x)))) for x in list(color)[:3])


def color_to_hex(color: Any) -> str:
    """A node's colorRGB -> '#RRGGBB' — the spelling a rule's color= takes."""
    if not color:
        return ""
    return "#" + "".join(f"{int(round(float(x))):02X}" for x in list(color)[:3])


_CSS_COLOR_FORMS = (
    'a CSS color string: "#RRGGBB", "rgb(r, g, b)" or "hsb(h, s%, b%)" -- '
    "or None (no quotes) for nodes with no color"
)


def parse_css_color(text: Any) -> str | None:
    """A CSS-style color literal -> the canonical 'r,g,b' matched against the tree.

    Accepted forms: '#RRGGBB'; 'rgb(r, g, b)' with integers 0-255; 'hsb(h, s%, b%)'
    with hue 0-360 and percentages 0-100 (converted via colorsys, each channel
    rounded with round()); the literal ``None``, passed through as the "has no
    color" token (``color=None`` selects nodes the CAD left uncolored)."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise RuleSyntaxError(f"A color literal is {_CSS_COLOR_FORMS}; got {type(text).__name__}")
    value = text.strip()
    hex_form = re.fullmatch(r"#([0-9a-fA-F]{6})", value)
    if hex_form:
        digits = hex_form.group(1)
        return ",".join(str(int(digits[i : i + 2], 16)) for i in (0, 2, 4))
    rgb_form = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", value)
    if rgb_form:
        channels = [int(g) for g in rgb_form.groups()]
        if max(channels) > 255:
            raise RuleSyntaxError(f"rgb() channels must be integers 0-255; got '{value}'")
        return ",".join(str(c) for c in channels)
    hsb_form = re.fullmatch(r"hsb\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)%\s*,\s*(\d+(?:\.\d+)?)%\s*\)", value)
    if hsb_form:
        hue, saturation, brightness = (float(g) for g in hsb_form.groups())
        if hue > 360 or saturation > 100 or brightness > 100:
            raise RuleSyntaxError(f"hsb() takes hue 0-360 and percentages 0-100; got '{value}'")
        rgb = colorsys.hsv_to_rgb(hue / 360, saturation / 100, brightness / 100)
        return ",".join(str(round(c * 255)) for c in rgb)
    raise RuleSyntaxError(f"A color literal is {_CSS_COLOR_FORMS}; got '{text}'")


# An empty glob on these fields says nothing: their text is never empty on a node that
# HAS the field, and a node without it matches no glob at all. Rejected statically
# rather than left as a silent no-op, since a user writing name="" means "unnamed".
_EMPTY_GLOB_MESSAGES = {
    "name": 'name == "" can never match and name != "" always matches, so neither says '
    'anything; an unnamed node is found with name != "*"',
    "type": 'type == "" can never match and type != "" always matches -- every node carries a type',
    "uuid": 'uuid == "" can never match and uuid != "" always matches, so neither says '
    'anything; a node carrying no uuid is found with uuid != "*"',
}


_EMPTY_ATTRIBUTES_MESSAGE = 'attributes={} says nothing -- name at least one key, e.g. attributes={"key": "glob"}'


def _reject_empty_glob(field: str, pattern: Any) -> None:
    if pattern == "" and field in _EMPTY_GLOB_MESSAGES:
        raise RuleSyntaxError(_EMPTY_GLOB_MESSAGES[field])


def _field_has_no_value(tree_node: dict[str, Any], field: str, attribute_key: str | None) -> bool:
    """Did the geometry leave this field unset on this node?

    The tree spells "no value" one way for every exporter (``TopologyTreeFormat.md``,
    envelope 2.0.0): key absent means the field does not apply to the node type, key
    present with ``null`` (``{}`` for attributes) means it applies but nothing is set,
    and ``""`` is never an unset sentinel. Both no-value states match nothing here.
    An attribute value of ``""`` and a material of ``{"name": ""}`` are present,
    merely empty, values.
    """
    if field == "attributes":
        return attribute_key not in tree_node.get("attributes", {})
    tree_key = "colorRGB" if field == "color" else field
    return tree_node.get(tree_key) is None


def _node_field_text(tree_node: dict[str, Any], field: str, attribute_key: str | None) -> str:
    """The text the glob is matched against -- only reached once the field HAS a value."""
    if field == "attributes":
        return str(tree_node["attributes"][attribute_key])
    if field == "color":
        return color_to_string(tree_node["colorRGB"])
    if field == "material":
        return str(tree_node["material"].get("name") or "")
    return str(tree_node[field])


# ---- predicates -----------------------------------------------------------------
class NodePredicate:
    """A boolean condition over one topology-tree node. Compose with ``&`` / ``|`` /
    ``~`` (parentheses required around comparisons — Python precedence)."""

    def matches(self, tree_node: dict[str, Any]) -> bool:
        raise NotImplementedError

    def to_ast(self) -> Any:
        """The engine-AST form of this predicate (for serialization)."""
        raise NotImplementedError

    def __and__(self, other: "NodePredicate") -> "NodePredicate":
        return _BooleanPredicate("BitAnd", self, _require_predicate(other))

    def __or__(self, other: "NodePredicate") -> "NodePredicate":
        return _BooleanPredicate("BitOr", self, _require_predicate(other))

    def __invert__(self) -> "NodePredicate":
        return _NotPredicate(self)

    def __bool__(self) -> bool:
        # RuleSyntaxError (a ValueError): one error type for every misuse of the rule
        # language, fluent or string. ValueError also matches the numpy/pandas
        # truthiness convention and stays pydantic-wrappable.
        raise RuleSyntaxError(
            "Predicates cannot be used with 'and'/'or'/'not' (Python cannot overload "
            "them); use '&' / '|' / '~' with parentheses around comparisons."
        )

    def __repr__(self) -> str:
        return _emit(self.to_ast(), 0)


def _require_predicate(value: Any) -> "NodePredicate":
    if not isinstance(value, NodePredicate):
        raise RuleSyntaxError(
            f"Predicates combine only with other predicates; got {type(value).__name__}. "
            'Remember the parentheses: (node.name == "a*") | (node.type == "Part")'
        )
    return value


class MatchPredicate(NodePredicate):
    """Glob comparison on one node field: ``node.<field> == "glob"`` (or ``!=``)."""

    def __init__(self, field: str, raw_value: Any, negate: bool = False, attribute_key: str | None = None) -> None:
        self.field = field
        self.raw_value = raw_value
        self.negate = negate
        self.attribute_key = attribute_key
        if field == "color":
            # A color literal normalizes to one exact 'r,g,b' text: compared by equality.
            self.pattern: str | None = parse_css_color(raw_value)
            self._glob: re.Pattern[str] | None = None
            return
        self.pattern = raw_value
        _reject_empty_glob(field, self.pattern)
        self._glob = _compile_glob_or_raise(field, self.pattern)

    def matches(self, tree_node: dict[str, Any]) -> bool:
        if self.pattern is None:
            # color=None asks "did the CAD leave this uncolored?" -- the one question
            # a glob cannot pose, so it inverts the no-value rule below.
            hit = _field_has_no_value(tree_node, self.field, self.attribute_key)
            return not hit if self.negate else hit
        if _field_has_no_value(tree_node, self.field, self.attribute_key):
            # A field with no value equals nothing, which is what makes == "*" read as
            # "has a value"; != is its exact complement, keeping ~(a == b) == (a != b).
            return self.negate
        text = _node_field_text(tree_node, self.field, self.attribute_key)
        hit = text == self.pattern if self._glob is None else self._glob.match(text) is not None
        return not hit if self.negate else hit

    def to_ast(self) -> Any:
        left: Any = NameNode(id=f"{NODE_NAME}.{self.field}")
        if self.field == "attributes":
            left = SubscriptNode(value=left, slice=ConstantNode(value=self.attribute_key), ctx="Load")
        right = ConstantNode(value=self.raw_value)
        return BinOpNode(op="NotEq" if self.negate else "Eq", left=left, right=right)

    def __invert__(self) -> "NodePredicate":
        return MatchPredicate(self.field, self.raw_value, not self.negate, self.attribute_key)


class _BooleanPredicate(NodePredicate):
    def __init__(self, op: str, left: NodePredicate, right: NodePredicate) -> None:
        self.op = op
        self.left = left
        self.right = right

    def matches(self, tree_node: dict[str, Any]) -> bool:
        if self.op == "BitAnd":
            return self.left.matches(tree_node) and self.right.matches(tree_node)
        return self.left.matches(tree_node) or self.right.matches(tree_node)

    def to_ast(self) -> Any:
        return BinOpNode(op=self.op, left=self.left.to_ast(), right=self.right.to_ast())


class _NotPredicate(NodePredicate):
    def __init__(self, operand: NodePredicate) -> None:
        self.operand = operand

    def matches(self, tree_node: dict[str, Any]) -> bool:
        return not self.operand.matches(tree_node)

    def to_ast(self) -> Any:
        return UnaryOpNode(op="Invert", operand=self.operand.to_ast())


# ---- the `node` proxy ----------------------------------------------------------
class _NodeFieldRef:
    """One node field awaiting comparison; ``==`` / ``!=`` build predicates."""

    def __init__(self, field: str, attribute_key: str | None = None) -> None:
        self._field = field
        self._attribute_key = attribute_key

    def _predicate(self, value: Any, negate: bool) -> MatchPredicate:
        # color literals are validated (and normalized) by the MatchPredicate itself
        if self._field != "color" and not isinstance(value, str):
            raise RuleSyntaxError(f"node.{self._field} compares to a string glob literal")
        return MatchPredicate(self._field, value, negate, self._attribute_key)

    def __eq__(self, value: Any) -> MatchPredicate:  # type: ignore[override]
        return self._predicate(value, negate=False)

    def __ne__(self, value: Any) -> MatchPredicate:  # type: ignore[override]
        return self._predicate(value, negate=True)

    __hash__ = None  # type: ignore[assignment]


class _NodeAttributesRef:
    def __getitem__(self, key: str) -> _NodeFieldRef:
        if not isinstance(key, str):
            raise RuleSyntaxError('node.attributes takes a string key: node.attributes["category/key"]')
        return _NodeFieldRef("attributes", key)


class _NodeProxy:
    """``node`` — the topology-tree node a navigation step is testing. Its fields
    build predicates under comparison: ``node.name == "*wing*"``.

    The fields are spelled out as properties (rather than routed through
    ``__getattr__``) so IDEs can autocomplete them — ``__getattr__`` remains only
    to give unknown fields a helpful error."""

    @property
    def name(self) -> _NodeFieldRef:
        """The node's name (glob on comparison: case-sensitive, whole string, `[seq]` allowed)."""
        return _NodeFieldRef("name")

    @property
    def type(self) -> _NodeFieldRef:
        """The node's type (e.g. Part, SolidBody; rejected on leaf steps)."""
        return _NodeFieldRef("type")

    @property
    def uuid(self) -> _NodeFieldRef:
        """The node's uuid (glob). Pins a selection to specific entities -- see the
        durability caveat on the uuid match field."""
        return _NodeFieldRef("uuid")

    @property
    def color(self) -> _NodeFieldRef:
        """The node's color; compare against a CSS color string
        ("#RRGGBB", "rgb(r, g, b)" or "hsb(h, s%, b%)")."""
        return _NodeFieldRef("color")

    @property
    def material(self) -> _NodeFieldRef:
        """The node's material name (glob)."""
        return _NodeFieldRef("material")

    @property
    def attributes(self) -> _NodeAttributesRef:
        """The node's CAD attribute map; subscribe a key: node.attributes["k"]."""
        return _NodeAttributesRef()

    def __getattr__(self, field: str) -> _NodeFieldRef:
        raise AttributeError(f"Unknown match field 'node.{field}'. Allowed: {', '.join(MATCH_FIELDS)}")

    def __repr__(self) -> str:
        return "node"


node = _NodeProxy()


# ---- kwargs sugar (detachable) ----------------------------------------------------
def match_fields_to_predicate(match_fields: dict[str, Any], method: str) -> NodePredicate | None:
    """Desugar keyword match fields into the predicate tree (AND of globs).

    This is the ONLY entry point for the kwargs syntax — deleting the sugar means
    deleting this function and the keyword acceptance at the call sites; the
    predicate tree stays the sole internal representation.
    """
    predicate: NodePredicate | None = None
    for key, value in match_fields.items():
        if key not in MATCH_FIELDS:
            raise RuleSyntaxError(_unknown_field_message(key))
        _reject_impossible_field(key, method)
        if key == "attributes":
            # the string route's own check, on the very AST this step serializes to
            _validate_kwarg_value(key, _literal_node(value))
            terms = [MatchPredicate("attributes", v, attribute_key=k) for k, v in value.items()]
        else:
            terms = [_NodeFieldRef(key)._predicate(value, negate=False)]  # noqa: SLF001
        for term in terms:
            predicate = term if predicate is None else predicate & term
    return predicate


def _unknown_field_message(key: str) -> str:
    hints = {
        "body": " (body=? use .bodies(name=...).faces(...))",
        "colorRGB": ' (colorRGB= became color=, taking a CSS string like "#FF0000")',
    }
    return f"Unknown match field '{key}'. Allowed: {', '.join(MATCH_FIELDS)}.{hints.get(key, '')}"


# Steps that pin exactly ONE node type, so `type=` on them says nothing new.
# `bodies` is deliberately absent: it matches a family of B-rep body subtypes, so
# `bodies(type="SolidBody")` genuinely narrows. `children`/`descendants` leave the
# type open, which is where `type=` earns its keep.
_SINGLE_TYPE_METHODS = ("model_files", "assemblies", "parts", "zones", *LEAF_METHODS)

# A match field that CANNOT exist on the node type a step names. Structural facts
# about what the node is -- not "the current exporter happens to omit it" -- so a
# rule using one can never match anything on any geometry, which makes it a static
# defect and a parse-time error (diagnostic policy: static defects raise).
_ABSENT_FIELDS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("model_files",),
        ("color", "material", "attributes"),
        "a model file is the on-disk wrapper, not a geometric entity -- these belong to the "
        "assemblies, parts, bodies and faces inside it",
    ),
    (
        ("faces",),
        ("material",),
        "a material belongs to the body, not to its surfaces -- match it one step up, "
        'e.g. bodies(material="steel").faces()',
    ),
    (
        ("solids", "boundaries", "zones"),
        ("color", "material", "attributes"),
        "these nodes come from the discrete formats (STL / UGRID / CGNS), which carry no color, "
        "no material and no attribute map",
    ),
)
_ABSENT_FIELD_REASON = {
    (method, field): reason for methods, fields, reason in _ABSENT_FIELDS for method in methods for field in fields
}


def _reject_impossible_field(field: str, method: str) -> None:
    """Reject a match field that cannot ever contribute on this step: redundant
    because the step already fixes the type, or absent from that node type."""
    if field == "type" and method in _SINGLE_TYPE_METHODS:
        raise RuleSyntaxError(f"'type' on a '{method}' step is redundant — it already fixes the node type")
    reason = _ABSENT_FIELD_REASON.get((method, field))
    if reason:
        raise RuleSyntaxError(f"'{field}' does not exist on a '{method}' step — {reason}")


def _predicate_fields(predicate: NodePredicate) -> set[str]:
    if isinstance(predicate, MatchPredicate):
        return {predicate.field}
    if isinstance(predicate, _BooleanPredicate):
        return _predicate_fields(predicate.left) | _predicate_fields(predicate.right)
    if isinstance(predicate, _NotPredicate):
        return _predicate_fields(predicate.operand)
    return set()


def ensure_predicate_valid_for_method(predicate: NodePredicate, method: str) -> None:
    """Step-level check for a live-built predicate argument."""
    _require_predicate(predicate)
    for field in sorted(_predicate_fields(predicate)):
        _reject_impossible_field(field, method)


# ---- validation of parsed rule ASTs ------------------------------------------------
def validate_rule_ast(ast_node: ExpressionNode) -> None:
    """Reject any AST outside the geometry-selection grammar (loudly, specifically)."""
    if isinstance(ast_node, UnaryOpNode) and ast_node.op == "Not":
        # `not` models fine, unlike `and`/`or`, so it arrives here rather than at
        # the engine boundary -- same wording, one lesson.
        raise RuleSyntaxError(_BOOLEAN_KEYWORDS_MESSAGE)
    if isinstance(ast_node, BinOpNode) and ast_node.op in _RULE_OPS:
        validate_rule_ast(ast_node.left)
        validate_rule_ast(ast_node.right)
        return
    if isinstance(ast_node, BinOpNode):
        if ast_node.op == "Add":
            raise RuleSyntaxError(_SELECTIONS_ARE_SETS)
        symbol = BINARY_OPERATORS[ast_node.op].symbol if ast_node.op in BINARY_OPERATORS else ast_node.op
        raise RuleSyntaxError(
            f"Operator '{symbol}' is not part of the geometry-selection grammar; "
            "allowed: '|' (union) and '-' (difference)"
        )
    _validate_chain(ast_node)


def _cannot_follow_message(method: str, terminal: str) -> str:
    return f"'{method}' cannot follow '{terminal}' — {'/'.join(LEAF_METHODS)} are terminal and end a chain"


def _validate_chain(ast_node: ExpressionNode) -> str | None:
    """Validate one navigation chain; return its terminal leaf method (None if open)."""
    if isinstance(ast_node, StaticCallNode):
        parts = ast_node.func_qualname.split(".")
        if len(parts) != 2 or parts[0] != ROOT_NAME:
            raise RuleSyntaxError(
                f"Rules must start from '{ROOT_NAME}' (e.g. {ROOT_NAME}.faces(...)); " f"got '{ast_node.func_qualname}'"
            )
        method = parts[1]
        _validate_navigation_call(method, ast_node)
        return method if method in LEAF_METHODS else None

    if isinstance(ast_node, CallOnResultNode):
        terminal = _validate_chain(ast_node.target)
        if terminal is not None:
            raise RuleSyntaxError(_cannot_follow_message(ast_node.method, terminal))
        _validate_navigation_call(ast_node.method, ast_node)
        return ast_node.method if ast_node.method in LEAF_METHODS else None

    if isinstance(ast_node, NameNode):
        if ast_node.id == ROOT_NAME:
            raise RuleSyntaxError(f"A selection needs at least one navigation step (e.g. {ROOT_NAME}.faces())")
        raise RuleSyntaxError(f"Unknown name '{ast_node.id}' — only '{ROOT_NAME}' is available")

    raise RuleSyntaxError(f"{type(ast_node).__name__} is not part of the geometry-selection grammar")


def _validate_navigation_call(method: str, ast_node: StaticCallNode | CallOnResultNode) -> None:
    if method not in NAV_METHODS:
        raise RuleSyntaxError(
            f"Unknown navigation method '{method}'. Allowed: "
            f"{', '.join(CONTAINER_METHODS)} (containers); "
            f"{', '.join(LEAF_METHODS)} (terminal)"
        )
    if ast_node.args and ast_node.kwargs:
        raise RuleSyntaxError(f"'{method}' takes either keyword match fields or one predicate expression, not both")
    if len(ast_node.args) > 1:
        raise RuleSyntaxError(f"'{method}' takes at most one predicate expression")
    if ast_node.args:
        _validate_predicate_ast(ast_node.args[0], method)
    for key, value in ast_node.kwargs.items():
        if key not in MATCH_FIELDS:
            raise RuleSyntaxError(_unknown_field_message(key))
        _reject_impossible_field(key, method)
        _validate_kwarg_value(key, value)


def _validate_predicate_ast(ast_node: ExpressionNode, method: str) -> None:
    if isinstance(ast_node, BinOpNode) and ast_node.op in _PREDICATE_BOOL_OPS:
        _validate_predicate_ast(ast_node.left, method)
        _validate_predicate_ast(ast_node.right, method)
        return
    if isinstance(ast_node, UnaryOpNode) and ast_node.op == "Invert":
        _validate_predicate_ast(ast_node.operand, method)
        return
    if isinstance(ast_node, BinOpNode) and ast_node.op in _COMPARE_OPS:
        _validate_field_comparison(ast_node, method)
        return
    raise RuleSyntaxError(
        "A predicate is comparisons on node.<field> combined with '&' / '|' / '~' "
        f"(with parentheses around comparisons); got {type(ast_node).__name__}"
    )


def _validate_field_comparison(comparison: BinOpNode, method: str) -> None:
    field = _field_ref_name(comparison.left)
    if field is None:
        raise RuleSyntaxError(
            "The left side of a predicate comparison must be node.<field> "
            f"({', '.join(f for f in MATCH_FIELDS if f != 'attributes')}) "
            'or node.attributes["key"]'
        )
    _reject_impossible_field(field, method)
    _validate_comparison_rhs(field, comparison.right)


def _field_ref_name(ast_node: ExpressionNode) -> str | None:
    """'name'/'type'/'color'/'material' for node.<field>; 'attributes' for
    node.attributes["key"]; None if not a valid field reference."""
    if isinstance(ast_node, NameNode) and ast_node.id.startswith(f"{NODE_NAME}."):
        field = ast_node.id.split(".", 1)[1]
        if field == "attributes":
            raise RuleSyntaxError('node.attributes needs a key: node.attributes["category/key"]')
        return field if field in MATCH_FIELDS else None
    if (
        isinstance(ast_node, SubscriptNode)
        and isinstance(ast_node.value, NameNode)
        and ast_node.value.id == f"{NODE_NAME}.attributes"
        and isinstance(ast_node.slice, ConstantNode)
        and isinstance(ast_node.slice.value, str)
    ):
        return "attributes"
    return None


def _string_literal(ast_node: ExpressionNode, message: str) -> str:
    if not isinstance(ast_node, ConstantNode) or not isinstance(ast_node.value, str):
        raise RuleSyntaxError(message)
    return ast_node.value


def _validate_kwarg_value(key: str, value: ExpressionNode) -> None:
    """The kwargs sugar: attributes= takes a dict literal; other fields take globs."""
    if key == "attributes":
        if not isinstance(value, DictNode):
            raise RuleSyntaxError('attributes= takes a dict literal, e.g. attributes={"key": "glob"}')
        if not value.keys:
            raise RuleSyntaxError(_EMPTY_ATTRIBUTES_MESSAGE)
        message = "attributes= entries must be string literals"
        for attribute_key in value.keys:
            _string_literal(attribute_key, message)
        for glob in value.values:
            _compile_glob_or_raise(key, _string_literal(glob, message))
        return
    _validate_glob_or_color(key, value)


def _validate_comparison_rhs(field: str, value: ExpressionNode) -> None:
    """The predicate form: node.attributes["k"] compares to a string glob; other
    fields as in the kwargs form."""
    if field == "attributes":
        glob = _string_literal(value, 'node.attributes["key"] compares to a string glob literal')
        _compile_glob_or_raise(field, glob)
        return
    _validate_glob_or_color(field, value)


def _validate_glob_or_color(key: str, value: ExpressionNode) -> None:
    if key == "color":
        if not isinstance(value, ConstantNode):
            raise RuleSyntaxError(f"color= takes {_CSS_COLOR_FORMS}")
        parse_css_color(value.value)
        return
    if isinstance(value, ListNode):
        raise RuleSyntaxError(f"'{key}=' takes a string glob literal")
    if not isinstance(value, ConstantNode):
        raise RuleSyntaxError(f"Match fields take literal values only; '{key}=' got {type(value).__name__}")
    if not isinstance(value.value, str):
        raise RuleSyntaxError(
            f"'{key}=' takes a string glob literal; got {type(value.value).__name__}"
            + (" (only color= takes None)" if value.value is None else "")
        )
    _reject_empty_glob(key, value.value)
    _compile_glob_or_raise(key, value.value)


# ---- parse entry point --------------------------------------------------------------
def parse_rule(source: str) -> ExpressionNode:
    """Parse a rule string into a validated AST (fresh empty context; no bindings)."""
    # Deferred imports: the parser pulls in ast/inspect (see expression/__init__ note).
    from flow360_schema.framework.expression.engine.eval_context import EvaluationContext
    from flow360_schema.framework.expression.engine.parser import expr_to_model
    from flow360_schema.framework.expression.engine.resolver import CallableResolver

    if not source.strip():
        raise RuleSyntaxError(_EMPTY_RULE_MESSAGE)
    try:
        model = expr_to_model(source, EvaluationContext(CallableResolver({}, {}, {}, ())))
    except SyntaxError as error:
        raise RuleSyntaxError(_malformed_source_message(source, error)) from error
    except ValueError as error:
        # The engine reports in its own terms (AST class names); re-diagnose in the
        # rule language's. Every misuse of the rule language is a RuleSyntaxError.
        raise RuleSyntaxError(_outside_grammar_message(source, error)) from error
    validate_rule_ast(model)
    # expr_to_model is declared as the base ExpressionNode while the engine's own
    # node fields are the concrete union, so the narrowing happens once, here.
    return _canonicalize_invert(cast(ExpressionNodeType, model))


_BOOLEAN_KEYWORDS_MESSAGE = (
    "Rules cannot use the 'and' / 'or' / 'not' keywords (Python cannot overload them); "
    "use '&' / '|' / '~' with parentheses around comparisons."
)


def _malformed_source_message(source: str, error: SyntaxError) -> str:
    """Python's own diagnosis is good ('(' was never closed); keep it, and add the
    one thing it cannot know -- that a rule was left ending on an operator."""
    trimmed = source.rstrip()
    if trimmed.endswith(("|", "-", "&", "~")):
        return f"A rule cannot end with '{trimmed[-1]}' -- the operator needs a right-hand side"
    return f"Not a valid rule expression: {error.msg}"


def _outside_grammar_message(source: str, error: ValueError) -> str:
    """Re-diagnose an engine rejection in the rule language's own words.

    Re-parsing here is free (this only runs on the error path) and, unlike matching
    the engine's message text, it cannot quietly stop working when the framework
    rewords its internals."""
    import ast  # deferred like the parser import above

    try:
        parsed: ast.expr = ast.parse(source, mode="eval").body
    except SyntaxError:
        if not ast.parse(source).body:  # nothing but comments
            return _EMPTY_RULE_MESSAGE
        return f"A rule is a single expression, not a statement; got {source!r}"
    if any(isinstance(node, ast.BoolOp) for node in ast.walk(parsed)):
        return _BOOLEAN_KEYWORDS_MESSAGE
    return f"Not part of the geometry-selection grammar: {error}"


def _canonicalize_invert(ast_node: ExpressionNodeType) -> ExpressionNodeType:
    """``~`` over a single comparison IS ``!=`` — collapse it, so a rule string and
    the same rule built fluently serialize to identical text (the fluent route
    already collapses it in ``MatchPredicate.__invert__``). ``~`` over a composite
    keeps its meaning. Runs after validation, so error messages still quote what
    the user wrote."""
    if isinstance(ast_node, UnaryOpNode) and ast_node.op == "Invert":
        operand = _canonicalize_invert(ast_node.operand)
        if isinstance(operand, BinOpNode) and operand.op in _COMPARE_OPS:
            return BinOpNode(op="NotEq" if operand.op == "Eq" else "Eq", left=operand.left, right=operand.right)
        return UnaryOpNode(op="Invert", operand=operand)
    if isinstance(ast_node, BinOpNode):
        left = _canonicalize_invert(ast_node.left)
        right = _canonicalize_invert(ast_node.right)
        return BinOpNode(op=ast_node.op, left=left, right=right)
    # A `~` can only occur inside a predicate, i.e. a call argument; kwargs and
    # subscripts carry literals only (see _validate_kwarg_value).
    if isinstance(ast_node, StaticCallNode):
        args = [_canonicalize_invert(arg) for arg in ast_node.args]
        return StaticCallNode(func_qualname=ast_node.func_qualname, args=args, kwargs=ast_node.kwargs)
    if isinstance(ast_node, CallOnResultNode):
        args = [_canonicalize_invert(arg) for arg in ast_node.args]
        target = _canonicalize_invert(ast_node.target)
        return CallOnResultNode(target=target, method=ast_node.method, args=args, kwargs=ast_node.kwargs)
    return ast_node


# ---- canonical serializer -------------------------------------------------------------
def rule_to_string(ast_node: ExpressionNode) -> str:
    """Canonical text form: double quotes, single space after commas, spaced operators,
    minimal parentheses. Byte-stable for semantically identical rules."""
    return _emit(ast_node, 0)


def _emit(ast_node: ExpressionNode, min_precedence: int) -> str:
    if isinstance(ast_node, BinOpNode):
        precedence = _PRECEDENCE[ast_node.op]
        symbol = BINARY_OPERATORS[ast_node.op].symbol
        # Left-associative: the right operand needs strictly higher precedence.
        left = _emit(ast_node.left, _operand_floor(ast_node, ast_node.left, precedence))
        right = _emit(ast_node.right, _operand_floor(ast_node, ast_node.right, precedence + 1))
        text = f"{left} {symbol} {right}"
        return f"({text})" if precedence < min_precedence else text
    if isinstance(ast_node, UnaryOpNode) and ast_node.op == "Invert":
        return f"~{_emit(ast_node.operand, _INVERT_PRECEDENCE)}"
    if isinstance(ast_node, StaticCallNode):
        return f"{ast_node.func_qualname}({_emit_call_args(ast_node)})"
    if isinstance(ast_node, CallOnResultNode):
        target = _emit(ast_node.target, 0)
        if isinstance(ast_node.target, BinOpNode):
            target = f"({target})"
        return f"{target}.{ast_node.method}({_emit_call_args(ast_node)})"
    if isinstance(ast_node, SubscriptNode):
        return f"{_emit(ast_node.value, 0)}[{_emit(ast_node.slice, 0)}]"
    if isinstance(ast_node, NameNode):
        return ast_node.id
    if isinstance(ast_node, ConstantNode):
        return _emit_literal(ast_node.value)
    if isinstance(ast_node, ListNode):
        return f"[{', '.join(_emit(e, 0) for e in ast_node.elements)}]"
    if isinstance(ast_node, DictNode):
        entries = ", ".join(
            f"{_emit(k, 0)}: {_emit(v, 0)}" for k, v in zip(ast_node.keys, ast_node.values, strict=True)
        )
        return f"{{{entries}}}"
    raise RuleSyntaxError(f"{type(ast_node).__name__} is not part of the geometry-selection grammar")


def _operand_floor(parent: BinOpNode, operand: ExpressionNodeType, floor: int) -> int:
    """Precedence floor an operand must clear to stay unparenthesized.

    Python's precedence makes `A | B - C` mean `A | (B - C)`, but reading it that
    way is not obvious, so the canonical form spells the grouping out whenever the
    two selection operators mix. Both spellings parse to the same tree; only the
    emitted text differs."""
    if parent.op not in _RULE_OPS or not isinstance(operand, BinOpNode):
        return floor
    if operand.op not in _RULE_OPS or operand.op == parent.op:
        return floor
    return _PRECEDENCE[operand.op] + 1


def _emit_call_args(ast_node: StaticCallNode | CallOnResultNode) -> str:
    parts = [_emit(arg, 0) for arg in ast_node.args]
    parts.extend(f"{k}={_emit(v, 0)}" for k, v in ast_node.kwargs.items())
    return ", ".join(parts)


def _emit_literal(value: Any) -> str:
    # json.dumps gives double-quoted, correctly escaped strings (valid Python too).
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


# ---- rule DESCRIPTIONS: the fluent authoring objects (pure program, NO tree) --
# A rule is an instance-agnostic description — navigation path + attribute
# filters. These objects hold ZERO geometry state; evaluation lives entirely in
# geometry/selection_evaluation.py (`resolve_selection(selection, tree)`), so a
# description can never leak or carry a stale geometry.
@dataclass(frozen=True)
class NavigationStep:
    """One navigation step of a rule. In-memory only — never serialized; the wire
    format is the rule string. ``match_fields`` records which surface syntax the
    user wrote (kwargs sugar vs predicate) for the string serializer; ``predicate``
    is the semantic form the evaluator reads."""

    method: str
    match_fields: dict[str, Any] | None
    predicate: NodePredicate | None


def _build_step_predicate(
    method: str, predicate_args: tuple[Any, ...], match_fields: dict[str, Any]
) -> NodePredicate | None:
    """One entry point for both condition syntaxes: keyword match fields (AND sugar,
    desugared by the grammar profile) or a single predicate expression."""
    if predicate_args and match_fields:
        raise RuleSyntaxError(f"'{method}' takes either keyword match fields or one predicate expression, not both")
    if len(predicate_args) > 1:
        raise RuleSyntaxError(f"'{method}' takes at most one predicate expression")
    if predicate_args:
        predicate: NodePredicate = predicate_args[0]
        ensure_predicate_valid_for_method(predicate, method)
        return predicate
    return match_fields_to_predicate(match_fields, method)


def _literal_node(value: Any) -> Any:
    """A raw match-field value (str/number/list/dict, as the user wrote it) -> engine AST."""
    if isinstance(value, (list, tuple)):
        return ListNode(elements=[ConstantNode(value=v) for v in value])
    if isinstance(value, dict):
        return DictNode(
            keys=[ConstantNode(value=k) for k in value],
            values=[ConstantNode(value=v) for v in value.values()],
        )
    return ConstantNode(value=value)


def _step_call_parts(step: NavigationStep) -> tuple[list[Any], dict[str, Any]]:
    """(args, kwargs) of one step's call, reproducing the surface syntax the user
    wrote: keyword match fields when recorded, else the predicate as the one arg."""
    if step.match_fields:
        return [], {key: _literal_node(value) for key, value in step.match_fields.items()}
    if step.predicate is not None:
        return [step.predicate.to_ast()], {}
    return [], {}


class RuleSelection:
    """Shared behavior of rule descriptions — a single chain (``SelectionChain``) or
    a lazy composition of chains (``CombinedSelection``): combinable with ``|`` /
    ``-`` (structure-preserving, so the composition stays serializable) and
    serializable to canonical rule text. Descriptions carry no geometry; evaluate
    one with ``selection_evaluation.resolve_selection(selection, tree)`` or through
    a draft."""

    def to_ast(self) -> Any:
        """The engine-AST form of this selection (for serialization)."""
        raise NotImplementedError

    def to_rule_string(self) -> str:
        """The canonical rule text denoting this selection. Note: predicates built
        live canonicalize ``~(x == y)`` to ``x != y`` (semantics-identical).

        This is also every selection's ``repr``: canonical rule text is valid client
        Python, so what prints is what you paste back."""
        return rule_to_string(self.to_ast())

    def _require_selection(self, other: Any, op: str) -> "RuleSelection":
        if not isinstance(other, RuleSelection):
            raise RuleSyntaxError(
                f"'{op}' combines rule descriptions with rule descriptions; got "
                f"{type(other).__name__}. Resolve the description first "
                "(selection_evaluation.resolve_selection) to combine plain id lists."
            )
        return other

    def __or__(self, other: Any) -> "CombinedSelection":
        return CombinedSelection("BitOr", self, self._require_selection(other, "|"))

    def __sub__(self, other: Any) -> "CombinedSelection":
        return CombinedSelection("Sub", self, self._require_selection(other, "-"))

    def __add__(self, other: Any) -> Any:
        raise RuleSyntaxError(_SELECTIONS_ARE_SETS)

    def __bool__(self) -> bool:
        # Without this, `sel_a and sel_b` short-circuits on default object truthiness
        # and silently returns one operand -- the classic pandas/numpy trap.
        raise RuleSyntaxError(
            "Selections cannot be used with 'and'/'or'/'not' or in boolean contexts "
            "(Python cannot overload them); combine selections with '|' (union) and "
            "'-' (difference)."
        )


class SelectionChain(RuleSelection):
    """A single RULE = a chain of navigation steps. Pure description: no geometry,
    no evaluation — kernel-legality checks happen at resolution, when a tree first
    exists.

    Each step takes EITHER keyword match fields (ANDed together) OR one predicate
    expression over ``node`` — never both. The fields, and what each accepts:

    - ``name`` — glob (case-sensitive, whole string; ``*``, ``?``, ``[seq]``, ``[!seq]``):
      ``"wing*"``, ``"*_face_?"``, ``"[fw]*_face_1"``
    - ``type`` — node type glob, only on the steps that do not already fix the type
      (``children``, ``descendants``, ``bodies``)
    - ``uuid`` — the node's own id (glob). This is the ESCAPE HATCH: it pins a rule to
      specific entities in one geometry rather than describing them, which is the one
      thing the rest of the language exists to avoid. A pinned rule does not survive
      re-import — the ids it names either resolve to different entities or stop
      resolving — and nothing detects that, so a hand-picked group has to be redone
      after the geometry changes. It matches the node's ``uuid`` key, not the entity
      id a resolved group reports (preview output, error examples, ``face_ids``,
      entity_info) — read the uuid from ``geometry.topology_tree``
    - ``color`` — CSS color string: ``"#FF0000"``, ``"rgb(255, 0, 0)"``, ``"hsb(0, 100%, 100%)"``;
      or ``None`` for nodes with no color
    - ``material`` — material-name glob: ``"*6061*"``
    - ``attributes`` — ``{"key": "glob"}``, one glob per attribute key

    A field the geometry left unset matches nothing, so ``"*"`` reads as "has a value"
    everywhere: ``name == "*"`` is "has a name", ``material == "*"`` is "has a
    material", ``{"finish": "*"}`` is "has a finish attribute". The complement finds
    the unset ones: ``name != "*"`` is "has no name". ``color`` takes no glob, so it
    asks the question the other way around: ``color=None`` is "has no color" and
    ``node.color != None`` is "has one".

    The predicate form compares the same fields and composes with ``&`` / ``|`` /
    ``~`` (parentheses required around comparisons)::

        root.bodies((node.material == "*6061*") & ~(node.name == "*fastener*")).faces()

    Whole selections compose as sets: ``|`` union, ``-`` difference::

        root.faces(node.name == "wall*") - root.faces(node.name == "*symmetry*")

    A step only offers the fields its node type can carry — a face has no material,
    a model file has no color — so the signatures differ per step by design."""

    def __init__(self, steps: list[NavigationStep] | None = None) -> None:
        self.steps = list(steps or [])

    def _add(self, method: str, match_fields: dict[str, Any], predicate: NodePredicate | None) -> "SelectionChain":
        # What validate_rule_ast refuses on the string route, the fluent route must refuse
        # too -- otherwise a chain serializes to text its own parser rejects, and resolves
        # with everything after the terminal step silently dropped.
        if self.steps and self.steps[-1].method in LEAF_METHODS:
            raise RuleSyntaxError(_cannot_follow_message(method, self.steps[-1].method))
        step = NavigationStep(method=method, match_fields=match_fields or None, predicate=predicate)
        return SelectionChain(self.steps + [step])

    @property
    def expr(self) -> dict[str, Any]:
        return {"steps": self.steps}

    # The navigation steps are spelled out one by one (rather than generated in a
    # loop) so IDEs can autocomplete the chain — these ARE the user-facing API.
    # The logic stays single-sourced in _add/_build_step_predicate; the grammar is
    # frozen at these ten methods, so the repetition does not grow.

    def children(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'children' step (direct children of the current scope).

        Example::

            root.children(node.name == "*.step").children((node.type == "Part") & ~(node.name == "*fixture*"))
        """
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return self._add("children", fields, _build_step_predicate("children", predicate, fields))

    def descendants(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'descendants' step (every node nested below the current scope).

        Example::

            root.descendants((node.type == "Part") | (node.type == "Assembly")).faces(node.color == "#FF0000")
        """
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return self._add("descendants", fields, _build_step_predicate("descendants", predicate, fields))

    def model_files(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'model_files' step (ModelFile nodes).

        Example::

            root.model_files((node.name == "*.step") | (node.name == "*.iges")).faces()  # also narrows the kernel check
        """
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return self._add("model_files", fields, _build_step_predicate("model_files", predicate, fields))

    def assemblies(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append an 'assemblies' step (Assembly nodes).

        Example::

            root.assemblies(node.name == "*wing*").bodies(~(node.name == "*fastener*")).faces()
        """
        fields = {
            **_given(name=name, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return self._add("assemblies", fields, _build_step_predicate("assemblies", predicate, fields))

    def parts(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'parts' step (Part nodes).

        Example::

            root.parts((node.attributes["__STEP/ProductName"] == "wing*") & ~(node.name == "*mirror*")).faces()
        """
        fields = {
            **_given(name=name, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return self._add("parts", fields, _build_step_predicate("parts", predicate, fields))

    def bodies(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'bodies' step (B-rep body nodes, any subtype).

        Example::

            root.bodies((node.material == "*6061*") | (node.color == "#C0C0C0")).faces()
        """
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return self._add("bodies", fields, _build_step_predicate("bodies", predicate, fields))

    def zones(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'zones' step (Zone nodes).

        Example::

            root.zones((node.name == "blk-1") | (node.name == "blk-2")).boundaries(~(node.name == "*interface*"))
        """
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return self._add("zones", fields, _build_step_predicate("zones", predicate, fields))

    def faces(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'faces' step (Face leaves). Terminal: no step may follow.

        Example::

            root.bodies(node.name == "wing*").faces((node.color == "#FF0000") & ~(node.name == "*fillet*"))
        """
        fields = {
            **_given(name=name, uuid=uuid, color=color, attributes=attributes),
            **match_fields,
        }
        return self._add("faces", fields, _build_step_predicate("faces", predicate, fields))

    def solids(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'solids' step (Solid leaves). Terminal: no step may follow.

        Example::

            root.model_files(node.name == "*.stl").solids((node.name == "shell*") | (node.name == "duct*"))
        """
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return self._add("solids", fields, _build_step_predicate("solids", predicate, fields))

    def boundaries(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> "SelectionChain":
        """Append a 'boundaries' step (Boundary leaves). Terminal: no step may follow.

        Example::

            root.boundaries(~(node.name == "*interface*") & ~(node.name == "*periodic*"))
        """
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return self._add("boundaries", fields, _build_step_predicate("boundaries", predicate, fields))

    def to_ast(self) -> Any:
        if not self.steps:
            raise RuleSyntaxError(f"A selection needs at least one navigation step (e.g. {ROOT_NAME}.faces())")
        args, kwargs = _step_call_parts(self.steps[0])
        current: Any = StaticCallNode(func_qualname=f"{ROOT_NAME}.{self.steps[0].method}", args=args, kwargs=kwargs)
        remaining = self.steps[1:]
        for step in remaining:
            args, kwargs = _step_call_parts(step)
            current = CallOnResultNode(target=current, method=step.method, args=args, kwargs=kwargs)
        return current

    def __repr__(self) -> str:
        if not self.steps:
            return f"{ROOT_NAME}.select()"
        return self.to_rule_string()


class CombinedSelection(RuleSelection):
    """Lazy ``|`` / ``-`` composition of rule descriptions. Keeps the composition
    STRUCTURE (rather than an id list) so a union-built group still serializes back
    to ``chain_a | chain_b``."""

    def __init__(self, op: str, left: RuleSelection, right: RuleSelection) -> None:
        self.op = op  # "BitOr" (union) or "Sub" (difference)
        self.left = left
        self.right = right

    def to_ast(self) -> Any:
        return BinOpNode(op=self.op, left=self.left.to_ast(), right=self.right.to_ast())

    def __repr__(self) -> str:
        return self.to_rule_string()


class _RootToken:
    """The module-level ``root`` token: the root of the geometry hierarchy tree as a
    pure name (same as the grammar's root, symmetric with ``node``). It carries no
    state at all — chains started here are instance-agnostic descriptions."""

    def __repr__(self) -> str:
        return ROOT_NAME

    def select(self) -> SelectionChain:
        """Start an empty rule (then chain steps)."""
        return SelectionChain()

    # Spelled out for IDE autocomplete — see the note on SelectionChain.

    def children(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'children' step (direct children of the tree root). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("children", fields, _build_step_predicate("children", predicate, fields))

    def descendants(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'descendants' step (every node in the tree). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("descendants", fields, _build_step_predicate("descendants", predicate, fields))

    def model_files(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'model_files' step (ModelFile nodes). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return SelectionChain()._add("model_files", fields, _build_step_predicate("model_files", predicate, fields))

    def assemblies(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with an 'assemblies' step (Assembly nodes). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("assemblies", fields, _build_step_predicate("assemblies", predicate, fields))

    def parts(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'parts' step (Part nodes). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("parts", fields, _build_step_predicate("parts", predicate, fields))

    def bodies(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        type: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        material: str = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'bodies' step (B-rep body nodes, any subtype). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, type=type, uuid=uuid, color=color, material=material, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("bodies", fields, _build_step_predicate("bodies", predicate, fields))

    def zones(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'zones' step (Zone nodes). Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return SelectionChain()._add("zones", fields, _build_step_predicate("zones", predicate, fields))

    def faces(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        color: str | None = _NOFILTER,
        attributes: dict[str, str] = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'faces' step (Face leaves). Terminal: no step may follow. Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid, color=color, attributes=attributes),
            **match_fields,
        }
        return SelectionChain()._add("faces", fields, _build_step_predicate("faces", predicate, fields))

    def solids(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'solids' step (Solid leaves). Terminal: no step may follow. Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return SelectionChain()._add("solids", fields, _build_step_predicate("solids", predicate, fields))

    def boundaries(
        self,
        *predicate: Any,
        name: str = _NOFILTER,
        uuid: str = _NOFILTER,
        **match_fields: Any,
    ) -> SelectionChain:
        """Start a rule with a 'boundaries' step (Boundary leaves). Terminal: no step may follow. Conditions as on
        :class:`SelectionChain` — see there for the full field table."""
        fields = {
            **_given(name=name, uuid=uuid),
            **match_fields,
        }
        return SelectionChain()._add("boundaries", fields, _build_step_predicate("boundaries", predicate, fields))


root = _RootToken()


# ---- rule string -> rule description (the program-walk bridge) -----------------
class _LocalBindingResolver(CallableResolver):
    """Resolves names from explicitly bound local objects ONLY (fresh per call —
    the no-global-context rule)."""

    def __init__(self, bindings: dict[str, Any]) -> None:
        super().__init__(callables={}, modules={}, imports={}, blacklist=())
        self._bindings = bindings

    def get_allowed_callable(self, qualname: str) -> Any:
        first, *rest = qualname.split(".")
        if first not in self._bindings:
            raise RuleSyntaxError(f"Callable '{qualname}' is not allowed")
        obj = self._bindings[first]
        for part in rest:
            obj = getattr(obj, part)
        return obj

    def can_evaluate(self, name: str) -> bool:
        return True


def rule_to_selection(rule_string: str) -> RuleSelection:
    """Parse + validate a rule string and rebuild it as a rule DESCRIPTION
    (``SelectionChain``/``CombinedSelection``) — still tree-agnostic. Engine
    evaluation dispatches through the same ``root``/``node`` objects fluent
    authoring uses, so the two routes cannot drift. Fresh, local context per call."""
    model = parse_rule(rule_string)
    bindings: dict[str, Any] = {ROOT_NAME: root, NODE_NAME: node}
    context = EvaluationContext(_LocalBindingResolver(bindings), initial_values=dict(bindings))
    selection: RuleSelection = model.evaluate(context)
    return selection
