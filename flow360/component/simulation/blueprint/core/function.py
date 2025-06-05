"""Data models and evaluator functions for full Python function definitions"""

from typing import Any

import pydantic as pd

from .context import EvaluationContext, ReturnValue
from .statements import StatementNodeType


class FunctionNode(pd.BaseModel):
    """
    Represents an entire function:
    def name(arg1, arg2, ...):
        <body...>
    """

    name: str
    args: list[str]
    defaults: dict[str, Any]
    body: list[StatementNodeType]

    def __call__(self, context: EvaluationContext, *call_args: Any) -> Any:
        # Add default values
        for arg_name, default_val in self.defaults.items():
            self.context.set(arg_name, default_val)

        # Add call arguments
        for arg_name, arg_val in zip(self.args, call_args, strict=False):
            self.context.set(arg_name, arg_val)

        try:
            for stmt in self.body:
                stmt.evaluate(self.context)
        except ReturnValue as rv:
            return rv.value

        return None
