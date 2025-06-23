"""Data models and evaluator functions for rvalue expression elements"""

import abc
from typing import Annotated, Any, Literal, Union

import pydantic as pd

from ..utils.operators import BINARY_OPERATORS, UNARY_OPERATORS
from .context import EvaluationContext
from .types import Evaluable

ExpressionNodeType = Annotated[
    # pylint: disable=duplicate-code
    Union[
        "NameNode",
        "ConstantNode",
        "BinOpNode",
        "RangeCallNode",
        "CallModelNode",
        "TupleNode",
        "ListNode",
        "ListCompNode",
        "SubscriptNode",
        "UnaryOpNode",
    ],
    pd.Field(discriminator="type"),
]


class ExpressionNode(pd.BaseModel, Evaluable, metaclass=abc.ABCMeta):
    """
    Base class for expressions (like `x > 3`, `range(n)`, etc.).

    Subclasses must implement the `evaluate` and `used_names` methods
    to support context-based evaluation and variable usage introspection.
    """

    def used_names(self) -> set[str]:
        """
        Return a set of variable names used by the expression.

        Returns:
            set[str]: A set of strings representing variable names used in the expression.
        """
        raise NotImplementedError


class NameNode(ExpressionNode):
    """
    Expression representing a name qualifier
    """

    type: Literal["Name"] = "Name"
    id: str

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        if raise_on_non_evaluable and not context.can_evaluate(self.id):
            raise ValueError(f"Name '{self.id}' cannot be evaluated at client runtime")
        if not force_evaluate and not context.can_evaluate(self.id):
            data_model = context.get_data_model(self.id)
            if data_model:
                return data_model.model_validate({"name": self.id, "value": context.get(self.id)})
            raise ValueError("Partially evaluable symbols need to possess a type annotation.")
        value = context.get(self.id)
        # Recursively evaluate if the returned value is evaluable
        if isinstance(value, Evaluable):
            value = value.evaluate(context, raise_on_non_evaluable, force_evaluate)
        return value

    def used_names(self) -> set[str]:
        return {self.id}


class ConstantNode(ExpressionNode):
    """
    Expression representing a constant numeric value
    """

    type: Literal["Constant"] = "Constant"
    value: Any

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:  # noqa: ARG002
        return self.value

    def used_names(self) -> set[str]:
        return set()


class UnaryOpNode(ExpressionNode):
    """
    Expression representing a unary operation
    """

    type: Literal["UnaryOp"] = "UnaryOp"
    op: str
    operand: "ExpressionNodeType"

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        operand_val = self.operand.evaluate(context, raise_on_non_evaluable, force_evaluate)

        if self.op not in UNARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {self.op}")

        return UNARY_OPERATORS[self.op](operand_val)

    def used_names(self) -> set[str]:
        return self.operand.used_names()


class BinOpNode(ExpressionNode):
    """
    Expression representing a binary operation
    """

    type: Literal["BinOp"] = "BinOp"
    left: "ExpressionNodeType"
    op: str
    right: "ExpressionNodeType"

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        left_val = self.left.evaluate(context, raise_on_non_evaluable, force_evaluate)
        right_val = self.right.evaluate(context, raise_on_non_evaluable, force_evaluate)

        if self.op not in BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {self.op}")

        return BINARY_OPERATORS[self.op](left_val, right_val)

    def used_names(self) -> set[str]:
        left = self.left.used_names()
        right = self.right.used_names()
        return left.union(right)


class SubscriptNode(ExpressionNode):
    """
    Expression representing an iterable object subscript
    """

    type: Literal["Subscript"] = "Subscript"
    value: "ExpressionNodeType"
    slice: "ExpressionNodeType"  # No proper slicing for now, only constants..
    ctx: str  # Only load context

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        value = self.value.evaluate(context, raise_on_non_evaluable, force_evaluate)
        item = self.slice.evaluate(context, raise_on_non_evaluable, force_evaluate)
        if self.ctx == "Load":
            if isinstance(item, float):
                item = int(item)
            return value[item]
        if self.ctx == "Store":
            raise NotImplementedError("Subscripted writes are not supported yet")

        raise ValueError(f"Invalid subscript context {self.ctx}")

    def used_names(self) -> set[str]:
        value = self.value.used_names()
        item = self.slice.used_names()
        return value.union(item)


class RangeCallNode(ExpressionNode):
    """
    Model for something like range(<expression>).
    """

    type: Literal["RangeCall"] = "RangeCall"
    arg: "ExpressionNodeType"

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> range:
        return range(self.arg.evaluate(context, raise_on_non_evaluable, force_evaluate))

    def used_names(self) -> set[str]:
        return self.arg.used_names()


class CallModelNode(ExpressionNode):
    """Represents a function or method call expression.

    This class handles both direct function calls and method calls through a fully qualified name.
    For example:
    - Simple function: "sum"
    - Method call: "np.array"
    - Nested attribute: "td.GridSpec.auto"
    """

    type: Literal["CallModel"] = "CallModel"
    func_qualname: str
    args: list["ExpressionNodeType"] = []
    kwargs: dict[str, "ExpressionNodeType"] = {}

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        try:
            # Split into parts for attribute traversal
            parts = self.func_qualname.split(".")

            if len(parts) == 1:
                # Direct function call
                func = context.resolve(parts[0])
            else:
                # Method or nested attribute call
                base = context.resolve(parts[0])

                # Traverse the attribute chain
                for part in parts[1:-1]:
                    base = getattr(base, part)

                # Get the final callable
                func = getattr(base, parts[-1])

            # Evaluate arguments
            args = [
                arg.evaluate(context, raise_on_non_evaluable, force_evaluate) for arg in self.args
            ]
            kwargs = {
                k: v.evaluate(context, raise_on_non_evaluable, force_evaluate)
                for k, v in self.kwargs.items()
            }

            return func(*args, **kwargs)

        except AttributeError as e:
            raise ValueError(
                f"Invalid attribute in call chain '{self.func_qualname}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error evaluating call to '{self.func_qualname}': {str(e)}") from e

    def used_names(self) -> set[str]:
        names = set()

        for arg in self.args:
            names = names.union(arg.used_names())

        for _, arg in self.kwargs.items():
            names = names.union(arg.used_names())

        return names


class TupleNode(ExpressionNode):
    """Model for tuple expressions."""

    type: Literal["Tuple"] = "Tuple"
    elements: list["ExpressionNodeType"]

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> tuple:
        return tuple(
            elem.evaluate(context, raise_on_non_evaluable, force_evaluate) for elem in self.elements
        )

    def used_names(self) -> set[str]:
        return self.arg.used_names()


class ListNode(ExpressionNode):
    """Model for list expressions."""

    type: Literal["List"] = "List"
    elements: list["ExpressionNodeType"]

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> list:
        return [
            elem.evaluate(context, raise_on_non_evaluable, force_evaluate) for elem in self.elements
        ]

    def used_names(self) -> set[str]:
        names = set()

        for arg in self.elements:
            names = names.union(arg.used_names())

        return names


class ListCompNode(ExpressionNode):
    """Model for list comprehension expressions."""

    type: Literal["ListComp"] = "ListComp"
    element: "ExpressionNodeType"  # The expression to evaluate for each item
    target: str  # The loop variable name
    iter: "ExpressionNodeType"  # The iterable expression

    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> list:
        result = []
        iterable = self.iter.evaluate(context, raise_on_non_evaluable, force_evaluate)
        for item in iterable:
            # Create a new context for each iteration with the target variable
            iter_context = context.copy()
            iter_context.set(self.target, item)
            result.append(
                self.element.evaluate(iter_context, raise_on_non_evaluable, force_evaluate)
            )
        return result

    def used_names(self) -> set[str]:
        element = self.element.used_names()
        iterable = self.iter.used_names()

        return element.union(iterable)
