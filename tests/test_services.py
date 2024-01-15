import json

import pytest

import flow360 as fl
from flow360 import services


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_validate_service():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
        "fluidProperties": {
            "modelType": "AirDensity",
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_name="SI"
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
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_name="SI"
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
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    }

    with pytest.raises(ValueError):
        errors, warning = services.validate_flow360_params_model(
            params_as_dict=params_data, unit_system_name=None
        )


def test_validate_service_incorrect_unit():
    params_data = {
        "geometry": {
            "refArea": {"units": "m", "value": 1.15315084119231},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_name="SI"
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
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_name="CGS"
    )

    assert errors[0]["msg"] == "ensure this value is greater than 0"


def test_validate_service_no_value():
    params_data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": None},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "boundaries": {},
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": None, "units": "m/s"}},
        "fluidProperties": {
            "modelType": "AirDensity",
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    errors, warning = services.validate_flow360_params_model(
        params_as_dict=params_data, unit_system_name="CGS"
    )

    assert errors[0]["msg"] == "field required"


def test_remove_dimensioned_type_none_leaves():
    data = {
        "geometry": {
            "refArea": {"units": "m**2", "value": None},
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "list": [
            {
                "refArea": {"units": "m**2", "value": None},
                "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            }
        ],
        "boundaries": {},
        "freestream": {"modelType": "FromVelocity", "velocity": {"value": None, "units": "m/s"}},
        "fluidProperties": {
            "modelType": "AirDensity",
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }
    expected = {
        "geometry": {
            "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            "meshUnit": {"units": "m", "value": 1.0},
        },
        "list": [
            {
                "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
            }
        ],
        "boundaries": {},
        "freestream": {"modelType": "FromVelocity"},
        "fluidProperties": {
            "modelType": "AirDensity",
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    processed = services.remove_dimensioned_type_none_leaves(data)
    assert processed == expected


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
            "modelType": "AirDensity",
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 1.225, "units": "kg/m**3"},
        },
    }

    with fl.CGS_unit_system:
        with pytest.raises(RuntimeError):
            errors, warning = services.validate_flow360_params_model(
                params_as_dict=params_data, unit_system_name="Imperial"
            )


def test_init_fork_with_update():
    with open("data/cases/case_13.json", "r") as fh:
        params = json.load(fh)

    data = services.get_default_fork(params)
    assert data


def test_init_fork_with_update_2():
    with open("data/cases/case_14_bet.json", "r") as fh:
        data = json.load(fh)

    params = services.get_default_fork(data)
    assert len(params.bet_disks) == 1
    assert params

    params_as_dict = services.params_to_dict(params)

    assert len(params_as_dict["BETDisks"]) == 1
    assert params_as_dict["BETDisks"][0]["thickness"] == 30.0
    assert params_as_dict["timeStepping"]["_addCFL"] == True
