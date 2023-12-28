import json

import pytest

import flow360 as fl
from flow360 import services


def test_validate_service():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
        "fluidProperties": {
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_context=fl.SI_unit_system
    )

    assert errors is None
    assert warning is None


def test_validate_service_missing_fluid_properties():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_context=fl.SI_unit_system
    )

    assert errors[0]["msg"] == "fluid_properties is required by freestream for unit conversion."


def test_validate_service_missing_unit_system():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    }

    with pytest.raises(ValueError):
        errors, warning = services.validate_flow360_params_model(
            params_as_dict=params_data, unit_system_context=None
        )


def test_validate_service_incorrect_unit():
    params_data = {
        "geometry": {
            "refArea": {"units": "m", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_context=fl.SI_unit_system
    )

    assert errors[0]["msg"] == "arg '1.15315084119231 m' does not match (length)**2"


def test_validate_service_incorrect_value():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": -1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_context=fl.SI_unit_system
    )

    assert errors[0]["msg"] == "ensure this value is greater than 0"


def test_validate_service_should_not_be_called_with_context():
    params_data = {
        "geometry": {
            "refArea": 1.15315084119231,
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
        "boundaries": {},
        "fluidProperties": {
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    with fl.CGS_unit_system:
        with pytest.raises(RuntimeError):
            errors, warning = services.validate_flow360_params_model(
                params_as_dict=params_data, unit_system_context=fl.SI_unit_system
            )
