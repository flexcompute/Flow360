"""Operator info for the parser module"""

# pylint: disable=too-few-public-methods

import operator
from collections.abc import Callable
from typing import Any, Union


class OpInfo:
    """Class to hold operator information."""

    def __init__(
        self, func: Union[Callable[[Any], Any], Callable[[Any, Any], Any]], symbol: str
    ) -> None:
        self.func = func
        self.symbol = symbol

    def __call__(self, *args: Any) -> Any:
        return self.func(*args)


UNARY_OPERATORS = {
    "UAdd": OpInfo(operator.pos, "+"),
    "USub": OpInfo(operator.neg, "-"),
}

BINARY_OPERATORS = {
    # Arithmetic operators
    "Add": OpInfo(operator.add, "+"),
    "Sub": OpInfo(operator.sub, "-"),
    "Mult": OpInfo(operator.mul, "*"),
    "Div": OpInfo(operator.truediv, "/"),
    "FloorDiv": OpInfo(operator.floordiv, "//"),
    "Mod": OpInfo(operator.mod, "%"),
    "Pow": OpInfo(operator.pow, "**"),
    # Comparison operators
    "Eq": OpInfo(operator.eq, "=="),
    "NotEq": OpInfo(operator.ne, "!="),
    "Lt": OpInfo(operator.lt, "<"),
    "LtE": OpInfo(operator.le, "<="),
    "Gt": OpInfo(operator.gt, ">"),
    "GtE": OpInfo(operator.ge, ">="),
    "Is": OpInfo(operator.is_, "is"),
    # Bitwise operators
    "BitAnd": OpInfo(operator.and_, "&"),
    "BitOr": OpInfo(operator.or_, "|"),
    "BitXor": OpInfo(operator.xor, "^"),
    "LShift": OpInfo(operator.lshift, "<<"),
    "RShift": OpInfo(operator.rshift, ">>"),
}
