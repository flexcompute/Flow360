from flow360.component.simulation.services import initialize_variable_space
from flow360.component.simulation.user_code.core.types import (
    get_referenced_expressions_and_user_variables,
)


def test_skip_variable_context_in_reference_collection():
    # Only variable_context contains an expression; it should be skipped entirely
    param_as_dict = {
        "private_attribute_asset_cache": {
            "variable_context": [
                {
                    "name": "vc_only",
                    "value": {"type_name": "expression", "expression": "1 + 2"},
                    "post_processing": False,
                    "description": None,
                    "metadata": None,
                }
            ]
        }
    }

    initialize_variable_space(param_as_dict, use_clear_context=True)

    expressions = get_referenced_expressions_and_user_variables(param_as_dict)
    assert expressions == []

    # If we add an expression outside of variable_context, it should be collected
    param_as_dict["some_field"] = {"type_name": "expression", "expression": "3 + 4"}
    expressions = get_referenced_expressions_and_user_variables(param_as_dict)
    assert sorted(expressions) == ["3 + 4"]
