import functools
import re
from typing import Any

from ..core.expressions import (
    BinOp,
    CallModel,
    Constant,
    List,
    ListComp,
    Name,
    RangeCall,
    Tuple,
    UnaryOp,
)
from ..core.function import Function
from ..core.statements import Assign, AugAssign, ForLoop, IfElse, Return, TupleUnpack
from ..utils.operators import BINARY_OPERATORS, UNARY_OPERATORS
from ..utils.types import TargetSyntax


def _indent(code: str, level: int = 1) -> str:
    """Add indentation to each line of code."""
    spaces = "    " * level
    return "\n".join(spaces + line if line else line for line in code.split("\n"))


def check_syntax_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            raise ValueError(
                f"Unsupported syntax type, available {[syntax.name for syntax in TargetSyntax]}"
            )
        return result

    return wrapper


@check_syntax_type
def _empty(syntax):
    if syntax == TargetSyntax.PYTHON:
        return "None"
    elif syntax == TargetSyntax.CPP:
        return "nullptr"


@check_syntax_type
def _name(expr, remap, name_filter):
    if name_filter and re.fullmatch(name_filter, expr.id):
        return ""
    else:
        return expr.id if expr.id not in remap else remap[expr.id]


@check_syntax_type
def _constant(expr):
    if isinstance(expr.value, str):
        return f"'{expr.value}'"
    return str(expr.value)


@check_syntax_type
def _unary_op(expr, syntax, remap, name_filter):
    op_info = UNARY_OPERATORS[expr.op]

    arg = expr_to_code(expr.operand, syntax, remap, name_filter)

    if not arg:
        return ""

    return f"{op_info.symbol}{arg}"


@check_syntax_type
def _binary_op(expr, syntax, remap, name_filter):
    left = expr_to_code(expr.left, syntax, remap, name_filter)
    right = expr_to_code(expr.right, syntax, remap, name_filter)

    allowed_operators = ["Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow"]

    if (not left or not right) and expr.op not in allowed_operators:
        raise ValueError(f"Operator {expr.op} arguments did not evaluate to a valid string")

    if not left and not right:
        return ""
    elif not right:
        return left
    elif not left:
        # Since those operations are not commutative
        # we cannot treat returning rvalue as neutral
        if expr.op in ["Mod", "Pow", "Sub", "Div", "FloorDiv"]:
            return ""
        return right

    if syntax == TargetSyntax.CPP:
        # Special case handling for operators not directly supported in CPP syntax, requires #include <cmath>
        if expr.op == "FloorDiv":
            return f"floor({left} / {right})"
        if expr.op == "Pow":
            return f"pow({left}, {right})"
        if expr.op == "Is":
            return f"&{left} == &{right}"

    op_info = BINARY_OPERATORS[expr.op]
    return f"({left} {op_info.symbol} {right})"


@check_syntax_type
def _range_call(expr, syntax, remap, name_filter):
    if syntax == TargetSyntax.PYTHON:
        arg = expr_to_code(expr.arg, syntax, remap, name_filter)
        if not arg:
            raise ValueError("Range call argument does not evaluate to a valid string")
        return f"range({arg})"

    raise ValueError("Range calls are only supported for Python target syntax")


@check_syntax_type
def _call_model(expr, syntax, remap, name_filter):
    if syntax == TargetSyntax.PYTHON:
        args = []
        for arg in expr.args:
            val_str = expr_to_code(arg, syntax, remap, name_filter)
            if not val_str:
                raise ValueError("Function argument did not evaluate to a valid string")
            args.append(val_str)
        args_str = ", ".join(args)
        kwargs_parts = []
        for k, v in expr.kwargs.items():
            if v is None:
                continue
            val_str = expr_to_code(v, syntax, remap, name_filter)
            if not val_str or val_str.isspace():
                continue
            kwargs_parts.append(f"{k}={val_str}")

        kwargs_str = ", ".join(kwargs_parts)
        all_args = ", ".join(x for x in [args_str, kwargs_str] if x)
        return f"{expr.func_qualname}({all_args})"
    elif syntax == TargetSyntax.CPP:
        args = []
        for arg in expr.args:
            val_str = expr_to_code(arg, syntax, remap, name_filter)
            if not val_str:
                raise ValueError("Function argument did not evaluate to a valid string")
            args.append(val_str)
        args_str = ", ".join(args)
        if expr.kwargs:
            raise ValueError("Named arguments are not supported in C++ syntax")
        return f"{expr.func_qualname}({args_str})"


@check_syntax_type
def _tuple(expr, syntax, remap, name_filter):
    elements = [expr_to_code(e, syntax, remap, name_filter) for e in expr.elements]
    if not all(elements):
        raise ValueError("List element did not evaluate to a valid string")

    if syntax == TargetSyntax.PYTHON:
        if len(expr.elements) == 0:
            return "()"
        elif len(expr.elements) == 1:
            return f"({elements[0]},)"
        return f"({', '.join(elements)})"
    elif syntax == TargetSyntax.CPP:
        if len(expr.elements) == 0:
            return "{}"
        elif len(expr.elements) == 1:
            return f"{{{elements[0]}}}"
        return f"{{{', '.join(elements)}}}"


@check_syntax_type
def _list(expr, syntax, remap, name_filter):
    elements = [expr_to_code(e, syntax, remap, name_filter) for e in expr.elements]
    if not all(elements):
        raise ValueError("List element did not evaluate to a valid string")

    if syntax == TargetSyntax.PYTHON:
        if len(expr.elements) == 0:
            return "[]"
        elements_str = ", ".join(elements)
        return f"[{elements_str}]"
    elif syntax == TargetSyntax.CPP:
        if len(expr.elements) == 0:
            return "{}"
        return f"{{{', '.join(elements)}}}"


def _list_comp(expr, syntax, remap, name_filter):
    if syntax == TargetSyntax.PYTHON:
        element = expr_to_code(expr.element, syntax, remap, name_filter)
        target = expr_to_code(expr.target, syntax, remap, name_filter)
        iterator = expr_to_code(expr.iter, syntax, remap, name_filter)

        if not element:
            raise ValueError("List comprehension element does not evaluate to a valid string")
        if not target:
            raise ValueError("List comprehension target does not evaluate to a valid string")
        if not iterator:
            raise ValueError("List comprehension iterator does not evaluate to a valid string")

        return f"[{element} for {target} in {iterator}]"

    raise ValueError("List comprehensions are only supported for Python target syntax")


def expr_to_code(
    expr: Any,
    syntax: TargetSyntax = TargetSyntax.PYTHON,
    remap: dict[str, str] = None,
    name_filter: str = None,
) -> str:
    """Convert an expression model back to source code."""
    if expr is None:
        return _empty(syntax)

    # Names and constants are language-agnostic (apart from symbol remaps)
    if isinstance(expr, Name):
        return _name(expr, remap, name_filter)

    elif isinstance(expr, Constant):
        return _constant(expr)

    elif isinstance(expr, UnaryOp):
        return _unary_op(expr, syntax, remap, name_filter)

    elif isinstance(expr, BinOp):
        return _binary_op(expr, syntax, remap, name_filter)

    elif isinstance(expr, RangeCall):
        return _range_call(expr, syntax, remap, name_filter)

    elif isinstance(expr, CallModel):
        return _call_model(expr, syntax, remap, name_filter)

    elif isinstance(expr, Tuple):
        return _tuple(expr, syntax, remap, name_filter)

    elif isinstance(expr, List):
        return _list(expr, syntax, remap, name_filter)

    elif isinstance(expr, ListComp):
        return _list_comp(expr, syntax, remap, name_filter)

    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")


def stmt_to_code(
    stmt: Any, syntax: TargetSyntax = TargetSyntax.PYTHON, remap: dict[str, str] = None
) -> str:
    if syntax == TargetSyntax.PYTHON:
        """Convert a statement model back to source code."""
        if isinstance(stmt, Assign):
            if stmt.target == "_":  # Expression statement
                return expr_to_code(stmt.value)
            return f"{stmt.target} = {expr_to_code(stmt.value, syntax, remap)}"

        elif isinstance(stmt, AugAssign):
            op_map = {
                "Add": "+=",
                "Sub": "-=",
                "Mult": "*=",
                "Div": "/=",
            }
            op_str = op_map.get(stmt.op, f"{stmt.op}=")
            return f"{stmt.target} {op_str} {expr_to_code(stmt.value, syntax, remap)}"

        elif isinstance(stmt, IfElse):
            code = [f"if {expr_to_code(stmt.condition)}:"]
            code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.body)))
            if stmt.orelse:
                code.append("else:")
                code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.orelse)))
            return "\n".join(code)

        elif isinstance(stmt, ForLoop):
            code = [f"for {stmt.target} in {expr_to_code(stmt.iter)}:"]
            code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.body)))
            return "\n".join(code)

        elif isinstance(stmt, Return):
            return f"return {expr_to_code(stmt.value, syntax, remap)}"

        elif isinstance(stmt, TupleUnpack):
            targets = ", ".join(stmt.targets)
            if len(stmt.values) == 1:
                # Single expression that evaluates to a tuple
                return f"{targets} = {expr_to_code(stmt.values[0], syntax, remap)}"
            else:
                # Multiple expressions
                values = ", ".join(expr_to_code(v, syntax, remap) for v in stmt.values)
                return f"{targets} = {values}"
        else:
            raise ValueError(f"Unsupported statement type: {type(stmt)}")

    raise NotImplementedError("Statement translation is not available for other syntax types yet")


def model_to_function(
    func: Function, syntax: TargetSyntax = TargetSyntax.PYTHON, remap: dict[str, str] = None
) -> str:
    if syntax == TargetSyntax.PYTHON:
        """Convert a Function model back to source code."""
        # Build the function signature
        args_with_defaults = []
        for arg in func.args:
            if arg in func.defaults:
                default_val = func.defaults[arg]
                if isinstance(default_val, (int, float, str, bool)):
                    args_with_defaults.append(f"{arg}={default_val}")
                else:
                    args_with_defaults.append(f"{arg}={expr_to_code(default_val, syntax, remap)}")
            else:
                args_with_defaults.append(arg)

        signature = f"def {func.name}({', '.join(args_with_defaults)}):"

        # Convert the function body
        body_lines = []
        for stmt in func.body:
            line = stmt_to_code(stmt)
            body_lines.append(line)

        body = "\n".join(body_lines) if body_lines else "pass"
        return f"{signature}\n{_indent(body)}"

    raise NotImplementedError("Function translation is not available for other syntax types yet")
