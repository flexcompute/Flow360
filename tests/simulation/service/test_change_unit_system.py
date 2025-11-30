import json
from pathlib import Path

import pytest

from flow360.component.simulation.services import change_unit_system


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_change_unit_system_with_nested_arrays():
    """Test that change_unit_system correctly handles nested arrays and numpy arrays."""
    # Load the simulation.json file from test data
    data_file = Path(__file__).parent / "data" / "simulation.json"

    assert data_file.exists(), f"Test data file not found: {data_file}"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify original unit system
    assert data["unit_system"]["name"] == "CGS"

    # Change unit system to SI
    converted_data = change_unit_system(data=data, target_unit_system="SI")

    # Verify unit system was updated
    assert converted_data["unit_system"]["name"] == "SI"

    # Verify specific conversions
    # moment_center: [0,0,0] cm -> [0.0, 0.0, 0.0] m
    moment_center = converted_data["reference_geometry"]["moment_center"]
    assert moment_center["value"] == [0.0, 0.0, 0.0]
    assert moment_center["units"] == "m"

    # moment_length: [99.99999999999999, ...] cm -> [0.9999999999999999, ...] m
    moment_length = converted_data["reference_geometry"]["moment_length"]
    expected_length = 99.99999999999999 / 100  # cm to m conversion
    assert abs(moment_length["value"][0] - expected_length) < 1e-10
    assert moment_length["units"] == "m"

    # Verify nested array conversion (profile_curve)
    nested_unyt_array = {
        "unit_system": {"name": "CGS"},
        "some_value": {"value": [[0, 0], [0, 1], [1, 0]], "units": "cm"},
    }
    converted = change_unit_system(data=nested_unyt_array, target_unit_system="SI")["some_value"]

    assert converted is not None, "profile_curve not found in test data"
    assert converted["units"] == "m"
    assert converted["value"] == [[0.0, 0.0], [0.0, 0.01], [0.01, 0.0]]
