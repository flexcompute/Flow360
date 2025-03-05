import pytest
from flow360.plugins.expressions.service import ExpressionRepr, ExpressionService, MOCK_DATA


def test_validate_and_evaluate_valid_expression():
    expr_repr = ExpressionRepr(name="a", expression="1.25 * m/s")
    service = ExpressionService(solver_variables={})
    
    result = service.validate_expression(expr_repr)
    assert result.errors in (None, [])
    assert result.expression_eval == MOCK_DATA[expr_repr.expression]

def test_validate_only_valid_expression():
    expr_repr = ExpressionRepr(name="a", expression="1.25 * m/s")
    service = ExpressionService(solver_variables={})
    
    result = service.validate_expression(expr_repr, evaluate=False)
    assert result.errors in (None, [])
    assert result.expression_eval is None

def test_validate_invalid_expression_no_raise():
    expr_repr = ExpressionRepr(name="b", expression="3 * km")
    service = ExpressionService(solver_variables={})
    
    result = service.validate_expression(expr_repr)
    assert result.errors == ["Invalid expression"]
    assert result.expression_eval is None

def test_validate_invalid_expression_with_raise():
    expr_repr = ExpressionRepr(name="b", expression="3 * km")
    service = ExpressionService(solver_variables={})
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_expression(expr_repr, raise_on_errors=True)
    assert "Invalid expression" in str(exc_info.value)

def test_validate_expressions_list():
    expr1 = ExpressionRepr(name="a", expression="1.25 * m/s")
    expr2 = ExpressionRepr(name="b", expression="3 * km +")  # Invalid expression.
    expr3 = ExpressionRepr(name="c", expression="MachRef + 1")
    service = ExpressionService(solver_variables={"MachRef": 0.85})
    
    results = service.validate_expressions([expr1, expr2, expr3])
    for res in results:
        if res.expression == "3 * km +":
            assert res.errors == ["Invalid expression"]
            assert res.expression_eval is None
        else:
            assert res.errors in (None, [])
            assert res.expression_eval == MOCK_DATA[res.expression]

def test_evaluate_expressions_list_no_evaluation():
    expr1 = ExpressionRepr(name="a", expression="1.25 * m/s")
    expr2 = ExpressionRepr(name="c", expression="MachRef + 1")
    service = ExpressionService(solver_variables={"MachRef": 0.85})
    
    results = service.validate_expressions([expr1, expr2], evaluate=False)
    for res in results:
        assert res.expression_eval is None
        assert res.errors in (None, [])

def test_validate_dict_single():
    data = {"name": "a", "expression": "1.25 * m/s"}
    service = ExpressionService(solver_variables={})
    result = service.validate_dict(data)
    assert result.errors in (None, [])
    assert result.expression == "1.25 * m/s"
    assert result.expression_eval == MOCK_DATA[result.expression]

def test_validate_dict_list():
    data = [
        {"name": "a", "expression": "1.25 * m/s"},
        {"name": "b", "expression": "3 * km +"},
        {"name": "c", "expression": "MachRef + 1"}
    ]
    service = ExpressionService(solver_variables={"MachRef": 0.85})
    results = service.validate_dict(data)
    for res in results:
        if res.expression == "3 * km +":
            assert res.errors == ["Invalid expression"]
            assert res.expression_eval is None
        else:
            assert res.errors in (None, [])
            assert res.expression_eval == MOCK_DATA[res.expression]

def test_validate_dict_invalid_input():
    service = ExpressionService(solver_variables={})
    with pytest.raises(ValueError):
        service.validate_dict("invalid input type")