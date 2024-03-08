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
    files = [
        "case_10.json",
        "case_13.json",
        "case_14_bet.json",
        "case_udd.json",
        "case_unsteady.json",
    ]

    for file in files:
        params = fl.Flow360Params(f"data/cases/{file}")
        assert params
        params.flow360_json()

    params = fl.Flow360Params(f"data/cases/case_5.json")
    assert params.turbulence_model_solver.reconstruction_gradient_limiter == 0.5
    params = fl.Flow360Params(f"data/cases/case_7.json")
    assert params.turbulence_model_solver.reconstruction_gradient_limiter == 1.0


def test_version_update():
    files = ["test_version_b16.json"]

    for file in files:
        params = fl.Flow360Params(f"data/cases/{file}")
        assert params


def test_updater_with_comments():
    file = "data/cases/case_comments_sliding_interfaces.json"

    params = fl.Flow360Params(file)

    assert params.fluid_properties.density == 1.225
    assert str(params.volume_zones["rotatingBlock-sphere1"].reference_frame.omega.units) == "rpm"
    assert float(params.volume_zones["rotatingBlock-sphere1"].reference_frame.omega.value) == 100
