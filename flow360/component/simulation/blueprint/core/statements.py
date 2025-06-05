"""Data models and evaluator functions for single-line Python statements"""

from typing import Annotated, Literal, Union

import pydantic as pd

from .context import EvaluationContext, ReturnValue
from .expressions import ExpressionNodeType
from .types import Evaluable

# Forward declaration of type
StatementNodeType = Annotated[
    # pylint: disable=duplicate-code
    Union[
        "AssignNode",
        "AugAssignNode",
        "IfElseNode",
        "ForLoopNode",
        "ReturnNode",
        "TupleUnpackNode",
    ],
    pd.Field(discriminator="type"),
]


class StatementNode(pd.BaseModel, Evaluable):
    """
    Base class for statements (like 'if', 'for', assignments, etc.).
    """

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        raise NotImplementedError


class AssignNode(StatementNode):
    """
    Represents something like 'result = <expr>'.
    """

    type: Literal["Assign"] = "Assign"
    target: str
    value: ExpressionNodeType

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        context.set(
            self.target, self.value.evaluate(context, raise_on_non_evaluable, force_evaluate)
        )


class AugAssignNode(StatementNode):
    """
    Represents something like 'result += <expr>'.
    The 'op' is again the operator class name (e.g. 'Add', 'Mult', etc.).
    """

    type: Literal["AugAssign"] = "AugAssign"
    target: str
    op: str
    value: ExpressionNodeType

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        old_val = context.get(self.target)
        increment = self.value.evaluate(context, raise_on_non_evaluable, force_evaluate)
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


class IfElseNode(StatementNode):
    """
    Represents an if/else block:
    if condition:
        <body...>
    else:
        <orelse...>
    """

    type: Literal["IfElse"] = "IfElse"
    condition: ExpressionNodeType
    body: list["StatementNodeType"]
    orelse: list["StatementNodeType"]

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        if self.condition.evaluate(context, raise_on_non_evaluable, force_evaluate):
            for stmt in self.body:
                stmt.evaluate(context, raise_on_non_evaluable, force_evaluate)
        else:
            for stmt in self.orelse:
                stmt.evaluate(context, raise_on_non_evaluable)


class ForLoopNode(StatementNode):
    """
    Represents a for loop:
    for <target> in <iter>:
        <body...>
    """

    type: Literal["ForLoop"] = "ForLoop"
    target: str
    iter: ExpressionNodeType
    body: list["StatementNodeType"]

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        iterable = self.iter.evaluate(context, raise_on_non_evaluable, force_evaluate)
        for item in iterable:
            context.set(self.target, item)
            for stmt in self.body:
                stmt.evaluate(context, raise_on_non_evaluable, force_evaluate)


class ReturnNode(StatementNode):
    """
    Represents a return statement: return <expr>.
    We'll raise a custom exception to stop execution in the function.
    """

    type: Literal["Return"] = "Return"
    value: ExpressionNodeType

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        val = self.value.evaluate(context, raise_on_non_evaluable, force_evaluate)
        raise ReturnValue(val)


class TupleUnpackNode(StatementNode):
    """Model for tuple unpacking assignments."""

    type: Literal["TupleUnpack"] = "TupleUnpack"
    targets: list[str]
    values: list[ExpressionNodeType]

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> None:
        evaluated_values = [
            val.evaluate(context, raise_on_non_evaluable, force_evaluate) for val in self.values
        ]
        for target, value in zip(self.targets, evaluated_values):
            context.set(target, value)
