"""Data models and evaluator functions for rvalue expression elements"""

import abc
from typing import Annotated, Any, Literal, Union

import pydantic as pd

from ..utils.operators import BINARY_OPERATORS, UNARY_OPERATORS
from .context import EvaluationContext
from .types import Evaluable

ExpressionType = Annotated[
    # pylint: disable=duplicate-code
    Union[
        "Name",
        "Constant",
        "BinOp",
        "RangeCall",
        "CallModel",
        "Tuple",
        "List",
        "ListComp",
        "Subscript",
    ],
    pd.Field(discriminator="type"),
]


class Expression(pd.BaseModel, Evaluable, metaclass=abc.ABCMeta):
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


class Name(Expression):
    """
    Expression representing a name qualifier
    """

    type: Literal["Name"] = "Name"
    id: str

    def evaluate(self, context: EvaluationContext, strict: bool, symbolic: bool = True) -> Any:
        # TODO: Stop hardcoding the symbolic flag
        if strict and not context.can_evaluate(self.id):
            raise ValueError(f"Name '{self.id}' cannot be evaluated at client runtime")
        value = context.get(self.id)
        print("get value from name: ", value)
        # Recursively evaluate if the returned value is evaluable
        if isinstance(value, Evaluable):
            print("IS an Evaluable, value type = ", type(value), " ", value, "|", self.id)
            value = value.evaluate(context, strict)
        else:
            from flow360.component.simulation.user_code import Variable

            # Very ugly implementation
            if self.id.startswith(("solution", "control")) and symbolic:
                return Variable(name=self.id, value=[1, 2, 3])

        return value

    def used_names(self) -> set[str]:
        return {self.id}


class Constant(Expression):
    """
    Expression representing a constant numeric value
    """

    type: Literal["Constant"] = "Constant"
    value: Any

    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:  # noqa: ARG002
        return self.value

    def used_names(self) -> set[str]:
        return set()


class UnaryOp(Expression):
    """
    Expression representing a unary operation
    """

    type: Literal["UnaryOp"] = "UnaryOp"
    op: str
    operand: "ExpressionType"

    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
        operand_val = self.operand.evaluate(context, strict)

        if self.op not in UNARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {self.op}")

        return UNARY_OPERATORS[self.op](operand_val)

    def used_names(self) -> set[str]:
        return self.operand.used_names()


class BinOp(Expression):
    """
    Expression representing a binary operation
    """

    type: Literal["BinOp"] = "BinOp"
    left: "ExpressionType"
    op: str
    right: "ExpressionType"

    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
        left_val = self.left.evaluate(context, strict)
        right_val = self.right.evaluate(context, strict)

        if self.op not in BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {self.op}")

        return BINARY_OPERATORS[self.op](left_val, right_val)

    def used_names(self) -> set[str]:
        left = self.left.used_names()
        right = self.right.used_names()
        return left.union(right)


class Subscript(Expression):
    """
    Expression representing an iterable object subscript
    """

    type: Literal["Subscript"] = "Subscript"
    value: "ExpressionType"
    slice: "ExpressionType"  # No proper slicing for now, only constants..
    ctx: str  # Only load context

    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
        value = self.value.evaluate(context, strict)
        item = self.slice.evaluate(context, strict)

        if self.ctx == "Load":
            return value[item]
        if self.ctx == "Store":
            raise NotImplementedError("Subscripted writes are not supported yet")

        raise ValueError(f"Invalid subscript context {self.ctx}")

    def used_names(self) -> set[str]:
        value = self.value.used_names()
        item = self.slice.used_names()
        return value.union(item)


class RangeCall(Expression):
    """
    Model for something like range(<expression>).
    """

    type: Literal["RangeCall"] = "RangeCall"
    arg: "ExpressionType"

    def evaluate(self, context: EvaluationContext, strict: bool) -> range:
        return range(self.arg.evaluate(context, strict))

    def used_names(self) -> set[str]:
        return self.arg.used_names()


class CallModel(Expression):
    """Represents a function or method call expression.

    This class handles both direct function calls and method calls through a fully qualified name.
    For example:
    - Simple function: "sum"
    - Method call: "np.array"
    - Nested attribute: "td.GridSpec.auto"
    """

    type: Literal["CallModel"] = "CallModel"
    func_qualname: str
    args: list["ExpressionType"] = []
    kwargs: dict[str, "ExpressionType"] = {}

    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
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
            args = [arg.evaluate(context, strict) for arg in self.args]
            kwargs = {k: v.evaluate(context, strict) for k, v in self.kwargs.items()}

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


class Tuple(Expression):
    """Model for tuple expressions."""

    type: Literal["Tuple"] = "Tuple"
    elements: list["ExpressionType"]

    def evaluate(self, context: EvaluationContext, strict: bool) -> tuple:
        return tuple(elem.evaluate(context, strict) for elem in self.elements)

    def used_names(self) -> set[str]:
        return self.arg.used_names()


class List(Expression):
    """Model for list expressions."""

    type: Literal["List"] = "List"
    elements: list["ExpressionType"]

    def evaluate(self, context: EvaluationContext, strict: bool) -> list:
        return [elem.evaluate(context, strict) for elem in self.elements]

    def used_names(self) -> set[str]:
        names = set()

        for arg in self.elements:
            names = names.union(arg.used_names())

        return names


class ListComp(Expression):
    """Model for list comprehension expressions."""

    type: Literal["ListComp"] = "ListComp"
    element: "ExpressionType"  # The expression to evaluate for each item
    target: str  # The loop variable name
    iter: "ExpressionType"  # The iterable expression

    def evaluate(self, context: EvaluationContext, strict: bool) -> list:
        result = []
        iterable = self.iter.evaluate(context, strict)
        for item in iterable:
            # Create a new context for each iteration with the target variable
            iter_context = context.copy()
            iter_context.set(self.target, item)
            result.append(self.element.evaluate(iter_context, strict))
        return result

    def used_names(self) -> set[str]:
        element = self.element.used_names()
        iterable = self.iter.used_names()

        return element.union(iterable)
