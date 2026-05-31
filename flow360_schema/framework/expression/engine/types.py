"""Shared type definitions for expression engine submodules"""

import abc
from enum import Enum
from typing import Any

from flow360_schema.framework.expression.engine.eval_context import EvaluationContext


class Evaluable(metaclass=abc.ABCMeta):
    """Base class for all classes that allow evaluation from their symbolic form"""

    @abc.abstractmethod
    def evaluate(
        self,
        context: EvaluationContext,
        raise_on_non_evaluable: bool = True,
        force_evaluate: bool = True,
    ) -> Any:
        """
        Evaluate the expression using the given context.

        Args:
            context (EvaluationContext): The context in which to evaluate the expression.
            raise_on_non_evaluable (bool): If True, raise an error on non-evaluable symbols;
                           if False, allow graceful failure or fallback behavior.
            force_evaluate (bool): If True, evaluate evaluable objects marked as
                          non-evaluable, instead of returning their identifier.
        Returns:
            Any: The evaluated value.
        """
        raise NotImplementedError


class TargetSyntax(Enum):
    """Target syntax enum, Python and"""

    PYTHON = "python"
    CPP = "cpp"
