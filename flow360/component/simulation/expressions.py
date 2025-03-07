from abc import abstractmethod

from pydantic_core import InitErrorDetails

from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.blueprint import expression_to_model

import pydantic as pd


class ExpressionVariable(Flow360BaseModel):
    name: str = pd.Field(None)



class ExpressionField(Flow360BaseModel):
    body: str = pd.Field("")

    _ctx: EvaluationContext = pd.PrivateAttr(EvaluationContext())

    @pd.model_validator(mode="before")
    def _validate_expression(cls, value):
        if not isinstance(value, str):
            details = InitErrorDetails(type="value_error", ctx={"error": "Input expression needs to be a string"})
            raise pd.ValidationError.from_exception_data("expression input type error", [details])
        try:
            _ = expression_to_model(value)
        except SyntaxError as s_err:
            details = InitErrorDetails(type="value_error", ctx={"error": s_err})
            raise pd.ValidationError.from_exception_data("expression syntax error", [details])
        except ValueError as v_err:
            details = InitErrorDetails(type="value_error", ctx={"error": v_err})
            raise pd.ValidationError.from_exception_data("expression value error", [details])

        return {"body": value}

    def evaluate(self) -> float:
        expr = expression_to_model(self.body)
        result = expr.evaluate(self._ctx)
        return result