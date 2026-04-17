import json
import os

from flow360.component.simulation.services import ValidationCalledBy, validate_model


def test_criterion_with_monitor_output_id():
    simulation_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "simulation_stopping_criterion_webui.json",
    )
    with open(simulation_path, "r") as file:
        data = json.load(file)

    _, errors, _ = validate_model(
        params_as_dict=data,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 0, "monitor_output"),
            "msg": "Value error, For stopping criterion setup, only one single `Point` entity is allowed in `ProbeOutput`/`SurfaceProbeOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 1, "monitor_output"),
            "msg": "Value error, The monitor field does not exist in the monitor output.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 2, "tolerance"),
            "msg": "Value error, The dimensions of monitor field and tolerance do not match.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 3, "monitor_output"),
            "msg": "Value error, The monitor output does not exist in the outputs list.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]
    assert len(errors) == len(expected_errors)
    for error, expected in zip(errors, expected_errors):
        assert error["loc"] == expected["loc"]
        assert error["type"] == expected["type"]
        assert error["msg"] == expected["msg"]
        assert error["ctx"]["relevant_for"] == expected["ctx"]["relevant_for"]
