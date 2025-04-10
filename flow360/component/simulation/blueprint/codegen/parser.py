import ast
import inspect
from collections.abc import Callable
from typing import Any

from ..core.context import EvaluationContext
from ..core.expressions import (
    BinOp,
    UnaryOp,
    CallModel,
    Constant,
    ListComp,
    Name,
    RangeCall,
    Tuple,
    Expression,
)
from ..core.expressions import (
    List as ListExpr,
)
from ..core.function import Function
from ..core.statements import (
    Assign,
    AugAssign,
    ForLoop,
    IfElse,
    Return,
    TupleUnpack,
)


def parse_expr(node: ast.AST, ctx: EvaluationContext) -> Any:
    """Parse a Python AST expression into our intermediate representation."""
    if isinstance(node, ast.Name):
        return Name(id=node.id)

    elif isinstance(node, ast.Constant):
        if hasattr(node, "value"):
            return Constant(value=node.value)
        else:
            return Constant(value=node.s)

    elif isinstance(node, ast.Attribute):
        # Handle attribute access (e.g., td.inf)
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            # Create a Name node with the full qualified name
            return Name(id=".".join(reversed(parts)))
        else:
            raise ValueError(f"Unsupported attribute access: {ast.dump(node)}")

    elif isinstance(node, ast.UnaryOp):
        return UnaryOp(op=type(node.op).__name__, operand=parse_expr(node.operand, ctx))

    elif isinstance(node, ast.BinOp):
        return BinOp(
            op=type(node.op).__name__,
            left=parse_expr(node.left, ctx),
            right=parse_expr(node.right, ctx),
        )

    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise ValueError("Only single comparisons are supported")
        return BinOp(
            op=type(node.ops[0]).__name__,
            left=parse_expr(node.left, ctx),
            right=parse_expr(node.comparators[0], ctx),
        )

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) == 1:
            return RangeCall(arg=parse_expr(node.args[0], ctx))

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
                raise ValueError(f"Unsupported function call: {ast.dump(node)}")
        else:
            raise ValueError(f"Unsupported function call: {ast.dump(node)}")

        # Parse arguments
        args = [parse_expr(arg, ctx) for arg in node.args]
        kwargs = {
            kw.arg: parse_expr(kw.value, ctx)
            for kw in node.keywords
            if kw.arg is not None and kw.value is not None  # Ensure value is not None
        }

        return CallModel(
            func_qualname=func_name,
            args=args,
            kwargs=kwargs,
        )

    elif isinstance(node, ast.Tuple):
        return Tuple(elements=[parse_expr(elt, ctx) for elt in node.elts])

    elif isinstance(node, ast.List):
        return ListExpr(elements=[parse_expr(elt, ctx) for elt in node.elts])

    elif isinstance(node, ast.ListComp):
        if len(node.generators) != 1:
            raise ValueError("Only single-generator list comprehensions are supported")
        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name):
            raise ValueError("Only simple targets in list comprehensions are supported")
        if gen.ifs:
            raise ValueError("If conditions in list comprehensions are not supported")
        return ListComp(
            element=parse_expr(node.elt, ctx),
            target=gen.target.id,
            iter=parse_expr(gen.iter, ctx),
        )

    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


def parse_stmt(node: ast.AST, ctx: EvaluationContext) -> Any:
    """Parse a Python AST statement into our intermediate representation."""
    if isinstance(node, ast.Assign):
        if len(node.targets) > 1:
            raise ValueError("Multiple assignment targets not supported")
        target = node.targets[0]

        if isinstance(target, ast.Name):
            return Assign(target=target.id, value=parse_expr(node.value, ctx))
        elif isinstance(target, ast.Tuple):
            if not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise ValueError("Only simple names supported in tuple unpacking")
            targets = [elt.id for elt in target.elts]
            if isinstance(node.value, ast.Tuple):
                values = [parse_expr(val, ctx) for val in node.value.elts]
                return TupleUnpack(targets=targets, values=values)
            else:
                return TupleUnpack(targets=targets, values=[parse_expr(node.value, ctx)])
        else:
            raise ValueError(f"Unsupported assignment target: {type(target)}")

    elif isinstance(node, ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple names supported in augmented assignment")
        return AugAssign(
            target=node.target.id,
            op=type(node.op).__name__,
            value=parse_expr(node.value, ctx),
        )

    elif isinstance(node, ast.Expr):
        # For expression statements, we use "_" as a dummy target
        return Assign(target="_", value=parse_expr(node.value, ctx))

    elif isinstance(node, ast.If):
        return IfElse(
            condition=parse_expr(node.test, ctx),
            body=[parse_stmt(stmt, ctx) for stmt in node.body],
            orelse=[parse_stmt(stmt, ctx) for stmt in node.orelse] if node.orelse else [],
        )

    elif isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple names supported as loop targets")
        return ForLoop(
            target=node.target.id,
            iter=parse_expr(node.iter, ctx),
            body=[parse_stmt(stmt, ctx) for stmt in node.body],
        )

    elif isinstance(node, ast.Return):
        if node.value is None:
            raise ValueError("Return statements must have a value")
        return Return(value=parse_expr(node.value, ctx))

    else:
        raise ValueError(f"Unsupported statement type: {type(node)}")


def function_to_model(
    source: str | Callable[..., Any],
    ctx: EvaluationContext | None = None,
) -> Function:
    """Parse a Python function definition into our intermediate representation.

    Args:
        source: Either a function object or a string containing the function definition
        ctx: Optional evaluation context
    """
    if ctx is None:
        ctx = EvaluationContext()

    # Convert function object to source string if needed
    if callable(source) and not isinstance(source, str):
        source = inspect.getsource(source)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # We expect a single function definition
    if (
        not isinstance(tree, ast.Module)
        or len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
    ):
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

    return Function(name=name, args=args, body=body, defaults=defaults)


def expression_to_model(
    source: str,
    ctx: EvaluationContext,
) -> Expression:
    """Parse a Python rvalue expression

    Args:
        source: a string containing the source
        ctx: Optional evaluation context
    """

    # Parse the source code into an AST
    tree = ast.parse(source)

    body = tree.body[0]

    # We expect a single line expression
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(body, ast.Expr):
        raise ValueError("Expected a single-line rvalue expression")

    expression = parse_expr(body.value, ctx)

    return expression
