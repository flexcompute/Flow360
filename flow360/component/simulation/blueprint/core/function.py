from typing import Any

import pydantic as pd

from .context import EvaluationContext, ReturnValue
from .statements import StatementType


class Function(pd.BaseModel):
    """
    Represents an entire function:
    def name(arg1, arg2, ...):
        <body...>
    """

    name: str
    args: list[str]
    defaults: dict[str, Any]
    body: list[StatementType]

    def __call__(self, *call_args: Any) -> Any:
        # Create empty context first
        context = EvaluationContext()

        # Add default values
        for arg_name, default_val in self.defaults.items():
            context.set(arg_name, default_val)

        # Add call arguments
        for arg_name, arg_val in zip(self.args, call_args, strict=False):
            context.set(arg_name, arg_val)

        try:
            for stmt in self.body:
                stmt.evaluate(context)
        except ReturnValue as rv:
            return rv.value

        return None
