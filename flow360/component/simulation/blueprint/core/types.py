import abc
from typing import Any

from .context import EvaluationContext


class Evaluable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
        pass
