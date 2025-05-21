"""Data models and evaluator functions for single-line Python statements"""

from typing import Annotated, Literal, Union

import pydantic as pd

from .context import EvaluationContext, ReturnValue
from .expressions import ExpressionType
from .types import Evaluable

# Forward declaration of type
StatementType = Annotated[
    # pylint: disable=duplicate-code
    Union[
        "Assign",
        "AugAssign",
        "IfElse",
        "ForLoop",
        "Return",
        "TupleUnpack",
    ],
    pd.Field(discriminator="type"),
]


class Statement(pd.BaseModel, Evaluable):
    """
    Base class for statements (like 'if', 'for', assignments, etc.).
    """

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        raise NotImplementedError


class Assign(Statement):
    """
    Represents something like 'result = <expr>'.
    """

    type: Literal["Assign"] = "Assign"
    target: str
    value: ExpressionType

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        context.set(self.target, self.value.evaluate(context, strict))


class AugAssign(Statement):
    """
    Represents something like 'result += <expr>'.
    The 'op' is again the operator class name (e.g. 'Add', 'Mult', etc.).
    """

    type: Literal["AugAssign"] = "AugAssign"
    target: str
    op: str
    value: ExpressionType

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        old_val = context.get(self.target)
        increment = self.value.evaluate(context, strict)
        if self.op == "Add":
            context.set(self.target, old_val + increment)
        elif self.op == "Sub":
            context.set(self.target, old_val - increment)
        elif self.op == "Mult":
            context.set(self.target, old_val * increment)
        elif self.op == "Div":
            context.set(self.target, old_val / increment)
        else:
            raise ValueError(f"Unsupported augmented assignment operator: {self.op}")


class IfElse(Statement):
    """
    Represents an if/else block:
    if condition:
        <body...>
    else:
        <orelse...>
    """

    type: Literal["IfElse"] = "IfElse"
    condition: ExpressionType
    body: list["StatementType"]
    orelse: list["StatementType"]

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        if self.condition.evaluate(context, strict):
            for stmt in self.body:
                stmt.evaluate(context, strict)
        else:
            for stmt in self.orelse:
                stmt.evaluate(context, strict)


class ForLoop(Statement):
    """
    Represents a for loop:
    for <target> in <iter>:
        <body...>
    """

    type: Literal["ForLoop"] = "ForLoop"
    target: str
    iter: ExpressionType
    body: list["StatementType"]

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        iterable = self.iter.evaluate(context, strict)
        for item in iterable:
            context.set(self.target, item)
            for stmt in self.body:
                stmt.evaluate(context, strict)


class Return(Statement):
    """
    Represents a return statement: return <expr>.
    We'll raise a custom exception to stop execution in the function.
    """

    type: Literal["Return"] = "Return"
    value: ExpressionType

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        val = self.value.evaluate(context, strict)
        raise ReturnValue(val)


class TupleUnpack(Statement):
    """Model for tuple unpacking assignments."""

    type: Literal["TupleUnpack"] = "TupleUnpack"
    targets: list[str]
    values: list[ExpressionType]

    def evaluate(self, context: EvaluationContext, strict: bool) -> None:
        evaluated_values = [val.evaluate(context, strict) for val in self.values]
        for target, value in zip(self.targets, evaluated_values):
            context.set(target, value)
