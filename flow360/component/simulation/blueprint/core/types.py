import abc
from .context import EvaluationContext
from typing import Any

class Evaluable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, context: EvaluationContext, strict: bool) -> Any:
        pass