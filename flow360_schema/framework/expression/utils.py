"""Utility functions for the expression engine."""

import ast
import re
from numbers import Number
from typing import Annotated, Any, NoReturn, cast

import numpy as np
import pydantic as pd
from pydantic.functional_validators import AfterValidator
from pydantic_core import InitErrorDetails
from unyt import unyt_array, unyt_quantity


def is_number_string(s: str) -> bool:
    """Check if the string represents a single scalar number"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def split_keep_delimiters(value: str, delimiters: list[str]) -> list[str]:
    """Split string but keep the delimiters"""
    escaped_delimiters = [re.escape(d) for d in delimiters]
    pattern = f"({'|'.join(escaped_delimiters)})"
    result = re.split(pattern, value)
    return [part for part in result if part != ""]


def convert_if_else(expression: str) -> str:
    """Convert ``if (...) ...; else ...;`` to ``? :`` syntax."""
    if expression.find("if") != -1:
        regex = r"\s*if\s*\(\s*(.*?)\s*\)\s*(.*?)\s*;\s*else\s*(.*?)\s*;\s*"
        subst = r"(\1) ? (\2) : (\3);"
        expression = re.sub(regex, subst, expression)
    return expression


def convert_caret_to_power(input_str: str) -> str:
    """Convert caret power syntax to ``powf()``."""
    enclosed = r"\([^(^)]+\)"
    non_negative_num = r"\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    number = r"[+-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    symbol = r"\b[a-zA-Z_][a-zA-Z_\d]*\b"
    base = rf"({enclosed}|{symbol}|{non_negative_num})"
    exponent = rf"({enclosed}|{symbol}|{number})"
    pattern = rf"{base}\s*\^\s*{exponent}"
    result = input_str
    while re.search(pattern, result):
        result = re.sub(pattern, r"powf(\1, \2)", result)
    return result


def convert_legacy_names(input_str: str) -> str:
    """Convert legacy variable names to current names."""
    old_names = ["rotMomentX", "rotMomentY", "rotMomentZ", "xyz"]
    new_names = ["momentX", "momentY", "momentZ", "coordinate"]
    result = input_str
    for old_name, new_name in zip(old_names, new_names, strict=True):
        pattern = r"\b(" + old_name + r")\b"
        result = re.sub(pattern, new_name, result)
    return result


def normalize_string_expression(expression: Any) -> Any:
    """Normalize a single string expression."""
    if not isinstance(expression, str):
        return expression
    expression = str(expression)
    expression = convert_if_else(expression)
    expression = convert_caret_to_power(expression)
    expression = convert_legacy_names(expression)
    return expression


def process_expressions(input_expressions: Any) -> Any:
    """Normalize a string expression or tuple of string expressions."""
    if isinstance(input_expressions, (str, float, int)):
        return normalize_string_expression(str(input_expressions))

    if isinstance(input_expressions, tuple):
        return tuple(normalize_string_expression(expression) for expression in input_expressions)
    return input_expressions


def handle_syntax_error(se: SyntaxError, source: str) -> NoReturn:
    """Handle expression syntax error."""
    caret = " " * (se.offset - 1) + "^" if se.text and se.offset else None
    msg = f"{se.msg} at line {se.lineno}, column {se.offset}"
    if caret and se.text:
        msg += f"\n{se.text.rstrip()}\n{caret}"

    raise pd.ValidationError.from_exception_data(
        "expression_syntax",
        [
            InitErrorDetails(
                type="value_error",
                input=source,
                ctx={
                    "line": se.lineno,
                    "column": se.offset,
                    "error": msg,
                },
            )
        ],
    )


def is_runtime_expression(value: Any) -> bool:
    """Check if the input value is a runtime expression (NaN-marked)."""
    if isinstance(value, unyt_quantity) and np.isnan(value.value):
        return True
    if isinstance(value, unyt_array) and np.isnan(value.value).any():
        return True
    if isinstance(value, Number) and np.isnan(float(cast(Any, value))):
        return True
    if isinstance(value, list) and any(np.isnan(float(cast(Any, item))) for item in value):
        return True
    return False


StringExpression = Annotated[str, AfterValidator(process_expressions)]


def validate_angle_expression_of_t_seconds(expr: str) -> list[str]:
    """Validate temporary rotation expressions that reference ``t_seconds``."""
    allowed_names = {
        "t_seconds",
        "t",
        "pi",
        "sin",
        "cos",
        "tan",
        "atan",
        "min",
        "max",
        "pow",
        "powf",
        "log",
        "exp",
        "sqrt",
        "abs",
        "ceil",
        "floor",
    }

    errors: list[str] = []

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        return [f"Syntax error in expression `{error.text}`: {error.msg}."]

    def is_valid_t_seconds_usage(ancestors: list[ast.AST]) -> bool:
        """Allow ``t_seconds`` only as a multiplicative factor."""
        if not ancestors:
            return False

        parent = ancestors[-1]
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mult):
            return True

        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
            if len(ancestors) >= 2:
                grandparent = ancestors[-2]
                if isinstance(grandparent, ast.BinOp) and isinstance(grandparent.op, (ast.Add, ast.Sub)):
                    return False
            return True

        return False

    def visit(node: ast.AST, ancestors: list[ast.AST] | None = None) -> None:
        if ancestors is None:
            ancestors = []

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                errors.append(f"Unexpected variable `{node.id}` found.")
            if node.id == "t_seconds" and not is_valid_t_seconds_usage(ancestors):
                errors.append(
                    "t_seconds must be used as a multiplicative factor, " "not directly added/subtracted with a number."
                )

        for child in ast.iter_child_nodes(node):
            visit(child, ancestors + [node])

    visit(tree)
    return errors
