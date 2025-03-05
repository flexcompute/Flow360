import pydantic as pd
import keyword
from typing import Union, Optional, List, Dict, Any, Tuple

from flow360.component.simulation.framework.base_model import Flow360BaseModel

class Unknown(Flow360BaseModel):
    reason: str = "solver variable"


class ExpressionEval(Flow360BaseModel):
    # The value can be a float, a 3D vector (represented as a tuple of three floats), or an Unknown, if depends on solver variables.
    value: Union[float, Tuple[float, float, float], Unknown]
    units: str

class ExpressionRepr(Flow360BaseModel):
    name: str
    expression: str
    expression_eval: Optional[ExpressionEval] = None
    errors: List[str] = []

    @pd.field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(
                f"'{v}' is not a valid Python identifier. "
                "A valid variable name must start with a letter or underscore and contain only letters, digits, or underscores."
            )
        if keyword.iskeyword(v):
            raise ValueError(
                f"'{v}' is a reserved keyword and cannot be used as a variable name."
            )
        return v



MOCK_DATA = {
    "1.25 * m/s": ExpressionEval(value=1.25, units="m/s"),
    "2 * a": ExpressionEval(value=2.5, units="m/s"),
    "MachRef + 1": ExpressionEval(value=Unknown(), units="dimensionless"),
}





class ExpressionService:
    """
    Service to process expression strings.
    
    The service provides both validation and evaluation methods:
    
    - validate_expression / validate_expressions:
      Validates an ExpressionRepr (or list thereof) by populating its errors field.
      Optionally, if evaluation is enabled (evaluate=True) and no errors are found,
      the expression is also evaluated.
      
      When raise_on_errors is True, a ValueError is raised if any validation errors occur.
    
    - validate_dict:
      Accepts a dict (or list of dicts), converts them into ExpressionRepr (or list thereof),
      and calls the appropriate validation method.
    
    Note: The actual parsing and evaluation logic is not implemented – a dummy evaluator is used.
    """
    def __init__(self, solver_variables: Optional[Dict[str, Any]] = None):
        self.solver_variables = solver_variables or {}

    def dummy_validate(self, expression_repr: ExpressionRepr) -> List[str]:
        """Dummy validation: returns an error if the expression is not found in MOCK_DATA."""
        if expression_repr.expression not in MOCK_DATA:
            return ["Invalid expression"]
        return []

    def dummy_evaluate(self, expression_repr: ExpressionRepr) -> ExpressionEval:
        """Dummy evaluation: returns the evaluation result from MOCK_DATA."""
        return MOCK_DATA[expression_repr.expression]

    def validate_expression(self, expression_repr: ExpressionRepr, raise_on_errors: bool = False, evaluate: bool = True) -> ExpressionRepr:
        """
        Validates (and optionally evaluates) a single ExpressionRepr.
        
        This method populates the errors field on the provided ExpressionRepr by using
        the dummy validation logic. If no errors are found and evaluate is True, the expression
        is evaluated using dummy_evaluate and the result is stored in expression_eval.
        
        Parameters:
            expression_repr: The ExpressionRepr to validate.
            raise_on_errors: If True, raises a ValueError when validation errors exist.
            evaluate: If True, performs evaluation after successful validation.
        
        Returns:
            The same ExpressionRepr with its errors field updated and (optionally) expression_eval set.
        
        Raises:
            ValueError: If raise_on_errors is True and validation errors exist.
        """
        expression_repr.errors = self.dummy_validate(expression_repr)
        if expression_repr.errors:
            if raise_on_errors:
                raise ValueError(f"Validation errors for '{expression_repr.name}': {expression_repr.errors}")
        else:
            if evaluate:
                expression_repr.expression_eval = self.dummy_evaluate(expression_repr)
        return expression_repr

    def validate_expressions(self, expressions: List[ExpressionRepr], raise_on_errors: bool = False, evaluate: bool = True) -> List[ExpressionRepr]:
        """
        Validates (and optionally evaluates) a list of ExpressionRepr instances.
        
        Parameters:
            expressions: List of ExpressionRepr objects to validate.
            raise_on_errors: If True, raises a ValueError when any validation error occurs.
            evaluate: If True, performs evaluation for expressions without errors.
        
        Returns:
            A list of ExpressionRepr objects with their errors field populated and (optionally) expression_eval set.
        """
        return [self.validate_expression(expr, raise_on_errors=raise_on_errors, evaluate=evaluate) for expr in expressions]

    def validate_dict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], raise_on_errors: bool = False, evaluate: bool = True) -> Union[ExpressionRepr, List[ExpressionRepr]]:
        """
        Parses a dict or list of dicts into ExpressionRepr (or list thereof) and validates them.
        
        Parameters:
            data: A dict representing a single expression or a list of dicts.
            raise_on_errors: If True, raises a ValueError when validation errors occur.
            evaluate: If True, performs evaluation for expressions without errors.
        
        Returns:
            - A validated ExpressionRepr if input data is a dict.
            - A list of validated ExpressionRepr if input data is a list of dicts.
        
        Raises:
            ValueError: If the input data is not a dict or a list of dicts.
        """
        if isinstance(data, dict):
            expr = ExpressionRepr.model_validate(data)
            return self.validate_expression(expr, raise_on_errors=raise_on_errors, evaluate=evaluate)
        elif isinstance(data, list):
            exprs = [ExpressionRepr.model_validate(item) for item in data]
            return self.validate_expressions(exprs, raise_on_errors=raise_on_errors, evaluate=evaluate)
        else:
            raise ValueError("Input data must be a dict or a list of dicts")


