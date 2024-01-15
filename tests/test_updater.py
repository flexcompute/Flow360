import json
import tempfile

import pytest

import flow360 as fl


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


data = {
    "geometry": {
        "refArea": 1.0,
        "momentCenter": [1, 2, 3],
        "momentLength": [37.490908, 20.362493146213485, 37.490908],
    },
    "boundaries": {
        "2": {"type": "SlipWall", "name": "symmetry"},
        "3": {"type": "Freestream", "name": "freestream"},
        "1": {
            "type": "NoSlipWall",
            "name": "wing",
            "Velocity": [0.0008162876014170267, 0.0016325752028340534, 0.0024488628042510802],
        },
    },
    "timeStepping": {
        "maxPseudoSteps": 500,
        "timeStepSize": 408352.8069698554,
        "CFL": {
            "type": "adaptive",
            "min": 0.1,
            "max": 10000.0,
            "maxRelativeChange": 1.0,
            "convergenceLimitingFactor": 0.25,
        },
    },
    "freestream": {
        "alphaAngle": 3.06,
        "betaAngle": 0.0,
        "Mach": 0.8404497144189705,
        "muRef": 4.292519319815164e-05,
        "Temperature": 288.15,
    },
    "volumeZones": {
        "zone1": {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "omegaRadians": 3.0773317581937964e-06,
                "centerOfRotation": [0.0, 0.0, 0.0],
                "axisOfRotation": [1, 0, 0],
            },
        },
        "zone3": {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "omegaRadians": 3.077331758193797e-06,
                "centerOfRotation": [0.0, 0.0, 0.0],
                "axisOfRotation": [1, 0, 0],
            },
        },
        "zone2": {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "omegaRadians": 3.0773317581937964e-06,
                "centerOfRotation": [0.0, 0.0, 0.0],
                "axisOfRotation": [1, 0, 0],
            },
        },
    },
}


def test_updater():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(data, temp_file)

    params = fl.Flow360Params(temp_file.name)
    print(params)


def test_updater_from_files():
    files = ["case_13.json", "case_14_bet.json"]

    for file in files:
        params = fl.Flow360Params(f"data/cases/{file}")
        assert params
