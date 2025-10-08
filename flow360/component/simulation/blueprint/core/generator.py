"""Code generator for the blueprint module, supports python and C++ syntax for now"""

# pylint: disable=too-many-return-statements

from typing import Any, Callable

from flow360.component.simulation.blueprint.core.expressions import (
    BinOpNode,
    CallModelNode,
    ConstantNode,
    ListCompNode,
    ListNode,
    NameNode,
    RangeCallNode,
    SubscriptNode,
    TupleNode,
    UnaryOpNode,
)
from flow360.component.simulation.blueprint.core.function import FunctionNode
from flow360.component.simulation.blueprint.core.statements import (
    AssignNode,
    AugAssignNode,
    ForLoopNode,
    IfElseNode,
    ReturnNode,
    TupleUnpackNode,
)
from flow360.component.simulation.blueprint.core.types import TargetSyntax
from flow360.component.simulation.blueprint.utils.operators import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
)


def _indent(code: str, level: int = 1) -> str:
    """Add indentation to each line of code."""
    spaces = "    " * level
    return "\n".join(spaces + line if line else line for line in code.split("\n"))


def _empty(syntax):
    if syntax == TargetSyntax.PYTHON:
        return "None"
    if syntax == TargetSyntax.CPP:
        return "nullptr"

    raise ValueError(
        f"Unsupported syntax type, available {[syntax.name for syntax in TargetSyntax]}"
    )


def _name(expr, name_translator):
    if name_translator:
        return name_translator(expr.id)
    return expr.id


def _constant(expr):
    if isinstance(expr.value, str):
        return f"'{expr.value}'"
    return str(expr.value)


def _unary_op(expr, syntax, name_translator):
    op_info = UNARY_OPERATORS[expr.op]

    arg = expr_to_code(expr.operand, syntax, name_translator)

    return f"{op_info.symbol}{arg}"


def _binary_op(expr, syntax, name_translator):
    left = expr_to_code(expr.left, syntax, name_translator)
    right = expr_to_code(expr.right, syntax, name_translator)

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


def _range_call(expr, syntax, name_translator):
    if syntax == TargetSyntax.PYTHON:
        arg = expr_to_code(expr.arg, syntax, name_translator)
        return f"range({arg})"

    raise ValueError("Range calls are only supported for Python target syntax")


def _call_model(expr, syntax, name_translator):
    if syntax == TargetSyntax.PYTHON:
        args = []
        for arg in expr.args:
            val_str = expr_to_code(arg, syntax, name_translator)
            args.append(val_str)
        args_str = ", ".join(args)
        kwargs_parts = []
        for k, v in expr.kwargs.items():
            if v is None:
                continue
            val_str = expr_to_code(v, syntax, name_translator)
            if not val_str or val_str.isspace():
                continue
            kwargs_parts.append(f"{k}={val_str}")

        kwargs_str = ", ".join(kwargs_parts)
        all_args = ", ".join(x for x in [args_str, kwargs_str] if x)
        return f"{expr.func_qualname}({all_args})"
    if syntax == TargetSyntax.CPP:
        args = []
        for arg in expr.args:
            val_str = expr_to_code(arg, syntax, name_translator)
            args.append(val_str)
        args_str = ", ".join(args)
        if expr.kwargs:
            raise ValueError("Named arguments are not supported in C++ syntax")
        return f"{name_translator(expr.func_qualname)}({args_str})"

    raise ValueError(
        f"Unsupported syntax type, available {[syntax.name for syntax in TargetSyntax]}"
    )


def _tuple(expr, syntax, name_translator):
    elements = [expr_to_code(e, syntax, name_translator) for e in expr.elements]

    if syntax == TargetSyntax.PYTHON:
        if len(expr.elements) == 0:
            return "()"
        if len(expr.elements) == 1:
            return f"({elements[0]},)"
        return f"({', '.join(elements)})"
    if syntax == TargetSyntax.CPP:
        if len(expr.elements) == 0:
            raise TypeError("Zero-length tuple is found in expression.")
        return f"std::vector<float>({{{', '.join(elements)}}})"

    raise ValueError(
        f"Unsupported syntax type, available {[syntax.name for syntax in TargetSyntax]}"
    )


def _list(expr, syntax, name_translator):
    elements = [expr_to_code(e, syntax, name_translator) for e in expr.elements]

    if syntax == TargetSyntax.PYTHON:
        if len(expr.elements) == 0:
            return "[]"
        elements_str = ", ".join(elements)
        return f"[{elements_str}]"
    if syntax == TargetSyntax.CPP:
        if len(expr.elements) == 0:
            raise TypeError("Zero-length list is found in expression.")

        return f"std::vector<float>({{{', '.join(elements)}}})"

    raise ValueError(
        f"Unsupported syntax type, available {[syntax.name for syntax in TargetSyntax]}"
    )


def _list_comp(expr, syntax, name_translator):
    if syntax == TargetSyntax.PYTHON:
        element = expr_to_code(expr.element, syntax, name_translator)
        target = expr_to_code(expr.target, syntax, name_translator)
        iterator = expr_to_code(expr.iter, syntax, name_translator)

        return f"[{element} for {target} in {iterator}]"

    raise ValueError("List comprehensions are only supported for Python target syntax")


def _subscript(expr, syntax, name_translator):  # pylint:disable=unused-argument
    return f"{name_translator(expr.value.id)}[{expr.slice.value}]"


def expr_to_code(
    expr: Any,
    syntax: TargetSyntax = TargetSyntax.PYTHON,
    name_translator: Callable[[str], str] = None,
) -> str:
    """Convert an expression model back to source code."""
    if expr is None:
        return _empty(syntax)

    # Names and constants are language-agnostic (apart from symbol remaps)
    if isinstance(expr, NameNode):
        return _name(expr, name_translator)

    if isinstance(expr, ConstantNode):
        return _constant(expr)

    if isinstance(expr, UnaryOpNode):
        return _unary_op(expr, syntax, name_translator)

    if isinstance(expr, BinOpNode):
        return _binary_op(expr, syntax, name_translator)

    if isinstance(expr, RangeCallNode):
        return _range_call(expr, syntax, name_translator)

    if isinstance(expr, CallModelNode):
        return _call_model(expr, syntax, name_translator)

    if isinstance(expr, TupleNode):
        return _tuple(expr, syntax, name_translator)

    if isinstance(expr, ListNode):
        return _list(expr, syntax, name_translator)

    if isinstance(expr, ListCompNode):
        return _list_comp(expr, syntax, name_translator)

    if isinstance(expr, SubscriptNode):
        return _subscript(expr, syntax, name_translator)

    raise ValueError(f"Unsupported expression type: {type(expr)}")


def stmt_to_code(
    stmt: Any, syntax: TargetSyntax = TargetSyntax.PYTHON, remap: dict[str, str] = None
) -> str:
    """Convert a statement model back to source code."""
    if syntax == TargetSyntax.PYTHON:
        if isinstance(stmt, AssignNode):
            if stmt.target == "_":  # Expression statement
                return expr_to_code(stmt.value)
            return f"{stmt.target} = {expr_to_code(stmt.value, syntax, remap)}"

        if isinstance(stmt, AugAssignNode):
            op_map = {
                "Add": "+=",
                "Sub": "-=",
                "Mult": "*=",
                "Div": "/=",
            }
            op_str = op_map.get(stmt.op, f"{stmt.op}=")
            return f"{stmt.target} {op_str} {expr_to_code(stmt.value, syntax, remap)}"

        if isinstance(stmt, IfElseNode):
            code = [f"if {expr_to_code(stmt.condition)}:"]
            code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.body)))
            if stmt.orelse:
                code.append("else:")
                code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.orelse)))
            return "\n".join(code)

        if isinstance(stmt, ForLoopNode):
            code = [f"for {stmt.target} in {expr_to_code(stmt.iter)}:"]
            code.append(_indent("\n".join(stmt_to_code(s, syntax, remap) for s in stmt.body)))
            return "\n".join(code)

        if isinstance(stmt, ReturnNode):
            return f"return {expr_to_code(stmt.value, syntax, remap)}"

        if isinstance(stmt, TupleUnpackNode):
            targets = ", ".join(stmt.targets)
            if len(stmt.values) == 1:
                # Single expression that evaluates to a tuple
                return f"{targets} = {expr_to_code(stmt.values[0], syntax, remap)}"
            # Multiple expressions
            values = ", ".join(expr_to_code(v, syntax, remap) for v in stmt.values)
            return f"{targets} = {values}"

        raise ValueError(f"Unsupported statement type: {type(stmt)}")

    raise NotImplementedError("Statement translation is not available for other syntax types yet")


def model_to_function(
    func: FunctionNode,
    syntax: TargetSyntax = TargetSyntax.PYTHON,
    remap: dict[str, str] = None,
) -> str:
    """Convert a Function model back to source code."""
    if syntax == TargetSyntax.PYTHON:
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
