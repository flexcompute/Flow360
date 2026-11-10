"""Python code parser using the AST module"""

import ast
import inspect
from collections.abc import Callable
from typing import Any

from flow360_schema.framework.expression.engine.ast_nodes import (
    BinOpNode,
    CallOnResultNode,
    ConstantNode,
    DictNode,
    ExpressionNode,
    ListCompNode,
    NameNode,
    RangeCallNode,
    StaticCallNode,
    SubscriptNode,
    TupleNode,
    UnaryOpNode,
)
from flow360_schema.framework.expression.engine.ast_nodes import ListNode as ListExpr
from flow360_schema.framework.expression.engine.eval_context import EvaluationContext
from flow360_schema.framework.expression.engine.function import FunctionNode
from flow360_schema.framework.expression.engine.statements import (
    AssignNode,
    AugAssignNode,
    ForLoopNode,
    IfElseNode,
    ReturnNode,
    TupleUnpackNode,
)


def _parse_keywords(call: ast.Call, ctx: EvaluationContext) -> dict[str, Any]:
    """A call's keyword arguments by name. CPython's AST spells a ``**mapping`` entry
    as a keyword with no name; skipping it would silently turn ``f(**opts)`` into
    ``f()``, so it is rejected like dict unpacking is."""
    kwargs: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("Keyword unpacking (**) is not supported")
        kwargs[keyword.arg] = parse_expr(keyword.value, ctx)
    return kwargs


def parse_expr(node: ast.AST, ctx: EvaluationContext) -> Any:
    """Parse a Python AST expression into our intermediate representation."""
    if isinstance(node, ast.Name):
        return NameNode(id=node.id)

    if isinstance(node, ast.Constant):
        if hasattr(node, "value"):
            return ConstantNode(value=node.value)
        return ConstantNode(value=node.s)

    if isinstance(node, ast.Attribute):
        # Handle attribute access (e.g., td.inf)
        parts = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            # Create a Name node with the full qualified name
            return NameNode(id=".".join(reversed(parts)))
        raise ValueError(f"Unsupported attribute access: {ast.dump(node)}")

    if isinstance(node, ast.UnaryOp):
        return UnaryOpNode(op=type(node.op).__name__, operand=parse_expr(node.operand, ctx))

    if isinstance(node, ast.BinOp):
        return BinOpNode(
            op=type(node.op).__name__,
            left=parse_expr(node.left, ctx),
            right=parse_expr(node.right, ctx),
        )

    if isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise ValueError("Only single comparisons are supported")
        return BinOpNode(
            op=type(node.ops[0]).__name__,
            left=parse_expr(node.left, ctx),
            right=parse_expr(node.comparators[0], ctx),
        )

    if isinstance(node, ast.Subscript):
        return SubscriptNode(
            value=parse_expr(node.value, ctx),
            slice=parse_expr(node.slice, ctx),
            ctx=type(node.ctx).__name__,
        )

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) == 1:
            return RangeCallNode(arg=parse_expr(node.args[0], ctx))

        # Build the full qualified name for the function
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle nested attributes (e.g., td.GridSpec.auto)
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                func_name = ".".join(reversed(parts))
            else:
                # The callee's base is not a dotted name but an arbitrary expression
                # (a call result like `root.bodies(...).faces(...)`, a parenthesized
                # operation, a subscript, ...). CallOnResultNode is defined for any such
                # target: evaluate it, then getattr the method on the result.
                return CallOnResultNode(
                    target=parse_expr(node.func.value, ctx),
                    method=node.func.attr,
                    args=[parse_expr(arg, ctx) for arg in node.args],
                    kwargs=_parse_keywords(node, ctx),
                )
        else:
            raise ValueError(f"Unsupported function call: {ast.dump(node)}")

        return StaticCallNode(
            func_qualname=func_name,
            args=[parse_expr(arg, ctx) for arg in node.args],
            kwargs=_parse_keywords(node, ctx),
        )

    if isinstance(node, ast.Tuple):
        return TupleNode(elements=[parse_expr(elt, ctx) for elt in node.elts])

    if isinstance(node, ast.List):
        return ListExpr(elements=[parse_expr(elt, ctx) for elt in node.elts])

    if isinstance(node, ast.Dict):
        # A `None` key is exactly how CPython's AST represents a `**expr` entry
        # (the unpacked expression sits in the parallel `values` slot) — the only
        # circumstance in which ast.Dict carries a None key.
        keys = [key for key in node.keys if key is not None]
        if len(keys) != len(node.keys):
            raise ValueError("Dict unpacking (**) is not supported")
        return DictNode(
            keys=[parse_expr(key, ctx) for key in keys],
            values=[parse_expr(value, ctx) for value in node.values],
        )

    if isinstance(node, ast.ListComp):
        if len(node.generators) != 1:
            raise ValueError("Only single-generator list comprehensions are supported")
        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name):
            raise ValueError("Only simple targets in list comprehensions are supported")
        if gen.ifs:
            raise ValueError("If conditions in list comprehensions are not supported")
        return ListCompNode(
            element=parse_expr(node.elt, ctx),
            target=gen.target.id,
            iter=parse_expr(gen.iter, ctx),
        )

    raise ValueError(f"Unsupported expression type: {type(node)}")


def parse_stmt(node: ast.AST, ctx: EvaluationContext) -> Any:
    """Parse a Python AST statement into our intermediate representation."""
    if isinstance(node, ast.Assign):
        if len(node.targets) > 1:
            raise ValueError("Multiple assignment targets not supported")
        target = node.targets[0]

        if isinstance(target, ast.Name):
            return AssignNode(target=target.id, value=parse_expr(node.value, ctx))
        if isinstance(target, ast.Tuple):
            if not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise ValueError("Only simple names supported in tuple unpacking")
            targets = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            if isinstance(node.value, ast.Tuple):
                values = [parse_expr(val, ctx) for val in node.value.elts]
                return TupleUnpackNode(targets=targets, values=values)
            return TupleUnpackNode(targets=targets, values=[parse_expr(node.value, ctx)])

        raise ValueError(f"Unsupported assignment target: {type(target)}")

    if isinstance(node, ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple names supported in augmented assignment")
        return AugAssignNode(
            target=node.target.id,
            op=type(node.op).__name__,
            value=parse_expr(node.value, ctx),
        )

    if isinstance(node, ast.Expr):
        # For expression statements, we use "_" as a dummy target
        return AssignNode(target="_", value=parse_expr(node.value, ctx))

    if isinstance(node, ast.If):
        return IfElseNode(
            condition=parse_expr(node.test, ctx),
            body=[parse_stmt(stmt, ctx) for stmt in node.body],
            orelse=[parse_stmt(stmt, ctx) for stmt in node.orelse] if node.orelse else [],
        )

    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple names supported as loop targets")
        return ForLoopNode(
            target=node.target.id,
            iter=parse_expr(node.iter, ctx),
            body=[parse_stmt(stmt, ctx) for stmt in node.body],
        )

    if isinstance(node, ast.Return):
        if node.value is None:
            raise ValueError("Return statements must have a value")
        return ReturnNode(value=parse_expr(node.value, ctx))

    raise ValueError(f"Unsupported statement type: {type(node)}")


def function_to_model(
    source: str | Callable[..., Any],
    ctx: EvaluationContext,
) -> FunctionNode:
    """Parse a Python function definition into our intermediate representation.

    Args:
        source: Either a function object or a string containing the function definition
        ctx: Evaluation context
    """

    # Convert function object to source string if needed
    if callable(source) and not isinstance(source, str):
        source = inspect.getsource(source)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # We expect a single function definition
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected a single function definition")

    func_def = tree.body[0]

    # Extract function name and arguments
    name = func_def.name
    args = [arg.arg for arg in func_def.args.args]
    defaults: dict[str, Any] = {}

    # Handle default values for arguments
    default_offset = len(func_def.args.args) - len(func_def.args.defaults)
    for i, default in enumerate(func_def.args.defaults):
        arg_name = func_def.args.args[i + default_offset].arg
        if isinstance(default, ast.Constant):
            defaults[arg_name] = default.value
        else:
            defaults[arg_name] = parse_expr(default, ctx)

    # Parse the function body
    body = [parse_stmt(stmt, ctx) for stmt in func_def.body]

    return FunctionNode(name=name, args=args, body=body, defaults=defaults)


def expr_to_model(
    source: str,
    ctx: EvaluationContext,
) -> ExpressionNode:
    """Parse a Python rvalue expression

    Args:
        source: a string containing the source
        ctx: Optional evaluation context
    """

    # Parse the source code into an AST
    tree = ast.parse(source)

    # We expect a single line expression; a comment-only source parses to an empty body
    body = tree.body[0] if len(tree.body) == 1 else None
    if not isinstance(body, ast.Expr):
        raise ValueError("Expected a single-line rvalue expression")

    return parse_expr(body.value, ctx)  # type: ignore[no-any-return]
