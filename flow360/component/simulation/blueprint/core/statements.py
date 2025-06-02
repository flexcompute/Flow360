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


class BlueprintStatement(pd.BaseModel, Evaluable):
    """
    Base class for statements (like 'if', 'for', assignments, etc.).
    """

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        raise NotImplementedError


class Assign(BlueprintStatement):
    """
    Represents something like 'result = <expr>'.
    """

    type: Literal["Assign"] = "Assign"
    target: str
    value: ExpressionType

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        context.set(self.target, self.value.evaluate(context, raise_error, force_evaluate))


class AugAssign(BlueprintStatement):
    """
    Represents something like 'result += <expr>'.
    The 'op' is again the operator class name (e.g. 'Add', 'Mult', etc.).
    """

    type: Literal["AugAssign"] = "AugAssign"
    target: str
    op: str
    value: ExpressionType

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        old_val = context.get(self.target)
        increment = self.value.evaluate(context, raise_error, force_evaluate)
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


class IfElse(BlueprintStatement):
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

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        if self.condition.evaluate(context, raise_error, force_evaluate):
            for stmt in self.body:
                stmt.evaluate(context, raise_error, force_evaluate)
        else:
            for stmt in self.orelse:
                stmt.evaluate(context, raise_error)


class ForLoop(BlueprintStatement):
    """
    Represents a for loop:
    for <target> in <iter>:
        <body...>
    """

    type: Literal["ForLoop"] = "ForLoop"
    target: str
    iter: ExpressionType
    body: list["StatementType"]

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        iterable = self.iter.evaluate(context, raise_error, force_evaluate)
        for item in iterable:
            context.set(self.target, item)
            for stmt in self.body:
                stmt.evaluate(context, raise_error, force_evaluate)


class Return(BlueprintStatement):
    """
    Represents a return statement: return <expr>.
    We'll raise a custom exception to stop execution in the function.
    """

    type: Literal["Return"] = "Return"
    value: ExpressionType

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        val = self.value.evaluate(context, raise_error, force_evaluate)
        raise ReturnValue(val)


class TupleUnpack(BlueprintStatement):
    """Model for tuple unpacking assignments."""

    type: Literal["TupleUnpack"] = "TupleUnpack"
    targets: list[str]
    values: list[ExpressionType]

    def evaluate(
        self, context: EvaluationContext, raise_error: bool, force_evaluate: bool = False
    ) -> None:
        evaluated_values = [
            val.evaluate(context, raise_error, force_evaluate, inlines) for val in self.values
        ]
        for target, value in zip(self.targets, evaluated_values):
            context.set(target, value)
