from pydantic import ValidationError

from flow360.plugins.expressions.service import ExpressionRepr





def test_valid_variable_names():
    valid_names = ["a", "result", "my_var", "_internal", "v2", "foo_bar", "x"]
    for name in valid_names:
        try:
            expr = ExpressionRepr(name=name, expression="1 * m/s")
            print(f"Valid variable name accepted: {name}")
        except ValidationError as e:
            print(f"Unexpected error for valid name '{name}': {e}")

def test_invalid_variable_names():
    # These names include ones that do not conform to identifier rules and reserved keywords.
    invalid_names = ["1invalid", "2foo", "if", "else", "while", "class", "for"]
    for name in invalid_names:
        try:
            expr = ExpressionRepr(name=name, expression="1 * m/s")
            print(f"Error: variable name '{name}' should have been rejected but was accepted.")
        except ValidationError as e:
            print(f"Correctly caught error for invalid name '{name}': {e}")
