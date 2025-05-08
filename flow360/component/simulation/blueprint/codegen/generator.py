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
)
from ..core.function import Function
from ..core.statements import Assign, AugAssign, ForLoop, IfElse, Return, TupleUnpack
from ..utils.operators import BINARY_OPERATORS


def _indent(code: str, level: int = 1) -> str:
    """Add indentation to each line of code."""
    spaces = "    " * level
    return "\n".join(spaces + line if line else line for line in code.split("\n"))


def expr_to_code(expr: Any) -> str:
    """Convert an expression model back to Python code."""
    if expr is None:
        return "None"

    if isinstance(expr, Name):
        return expr.id

    elif isinstance(expr, Constant):
        if isinstance(expr.value, str):
            return f"'{expr.value}'"
        return str(expr.value)

    elif isinstance(expr, BinOp):
        op_info = BINARY_OPERATORS[expr.op]
        return f"({expr_to_code(expr.left)} {op_info.symbol} {expr_to_code(expr.right)})"

    elif isinstance(expr, RangeCall):
        return f"range({expr_to_code(expr.arg)})"

    elif isinstance(expr, CallModel):
        args_str = ", ".join(expr_to_code(arg) for arg in expr.args)
        kwargs_parts = []
        for k, v in expr.kwargs.items():
            if v is None:
                continue
            val_str = expr_to_code(v)
            if not val_str or val_str.isspace():
                continue
            kwargs_parts.append(f"{k}={val_str}")

        kwargs_str = ", ".join(kwargs_parts)
        all_args = ", ".join(x for x in [args_str, kwargs_str] if x)
        return f"{expr.func_qualname}({all_args})"

    elif isinstance(expr, Tuple):
        if len(expr.elements) == 0:
            return "()"
        elif len(expr.elements) == 1:
            return f"({expr_to_code(expr.elements[0])},)"
        return f"({', '.join(expr_to_code(e) for e in expr.elements)})"

    elif isinstance(expr, List):
        if not expr.elements:
            return "[]"
        elements = [expr_to_code(e) for e in expr.elements]
        elements_str = ", ".join(elements)
        return f"[{elements_str}]"

    elif isinstance(expr, ListComp):
        return f"[{expr_to_code(expr.element)} for {expr.target} in {expr_to_code(expr.iter)}]"

    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")


def stmt_to_code(stmt: Any) -> str:
    """Convert a statement model back to Python code."""
    if isinstance(stmt, Assign):
        if stmt.target == "_":  # Expression statement
            return expr_to_code(stmt.value)
        return f"{stmt.target} = {expr_to_code(stmt.value)}"

    elif isinstance(stmt, AugAssign):
        op_map = {
            "Add": "+=",
            "Sub": "-=",
            "Mult": "*=",
            "Div": "/=",
        }
        op_str = op_map.get(stmt.op, f"{stmt.op}=")
        return f"{stmt.target} {op_str} {expr_to_code(stmt.value)}"

    elif isinstance(stmt, IfElse):
        code = [f"if {expr_to_code(stmt.condition)}:"]
        code.append(_indent("\n".join(stmt_to_code(s) for s in stmt.body)))
        if stmt.orelse:
            code.append("else:")
            code.append(_indent("\n".join(stmt_to_code(s) for s in stmt.orelse)))
        return "\n".join(code)

    elif isinstance(stmt, ForLoop):
        code = [f"for {stmt.target} in {expr_to_code(stmt.iter)}:"]
        code.append(_indent("\n".join(stmt_to_code(s) for s in stmt.body)))
        return "\n".join(code)

    elif isinstance(stmt, Return):
        return f"return {expr_to_code(stmt.value)}"

    elif isinstance(stmt, TupleUnpack):
        targets = ", ".join(stmt.targets)
        if len(stmt.values) == 1:
            # Single expression that evaluates to a tuple
            return f"{targets} = {expr_to_code(stmt.values[0])}"
        else:
            # Multiple expressions
            values = ", ".join(expr_to_code(v) for v in stmt.values)
            return f"{targets} = {values}"

    else:
        raise ValueError(f"Unsupported statement type: {type(stmt)}")


def model_to_function(func: Function) -> str:
    """Convert a Function model back to Python code."""
    # Build the function signature
    args_with_defaults = []
    for arg in func.args:
        if arg in func.defaults:
            default_val = func.defaults[arg]
            if isinstance(default_val, int | float | bool | str):
                args_with_defaults.append(f"{arg}={default_val}")
            else:
                args_with_defaults.append(f"{arg}={expr_to_code(default_val)}")
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
