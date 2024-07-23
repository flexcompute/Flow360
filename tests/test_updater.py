import json
import tempfile

import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_legacy import LinearSolverLegacy
from flow360.component.flow360_params.initial_condition import (
    ExpressionInitialCondition,
)
from flow360.component.flow360_params.updater import (
    UPDATE_MAP,
    _find_update_path,
    _no_update,
)
from flow360.exceptions import Flow360NotImplementedError


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

data_turbulence = {
    "geometry": {
        "refArea": 1.0,
        "momentCenter": [1, 2, 3],
        "momentLength": [37.490908, 20.362493146213485, 37.490908],
    },
    "boundaries": {
        "2": {
            "type": "SubsonicInflow",
            "name": "test",
            "totalPressureRatio": 1.0,
            "totalTemperatureRatio": 1.0,
            "turbulenceQuantities": {"modifiedTurbulentViscosity": 1.0},
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
        "turbulenceQuantities": {"modifiedTurbulentViscosity": 1.0},
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
        "case_18.json",
        "case_14_bet.json",
        "case_18.json",
        "case_udd.json",
        "case_udd_legacy.json",
        "case_unsteady.json",
        "case_customDynamics1.json",
        "case_HeatTransfer.json",
        "case_20.json",
        "case_boundaries.json",
    ]

    for file in files:
        params = fl.Flow360Params(f"data/cases/{file}")
        assert params
        params.flow360_json()

    params = fl.Flow360Params("data/cases/case_5.json")
    assert params.turbulence_model_solver.reconstruction_gradient_limiter == 0.5
    assert params.initial_condition == ExpressionInitialCondition(
        constants={"gamma": "1.4"}, rho=1, u="x", v=0, w=0, p="1/gamma"
    )
    params = fl.Flow360Params("data/cases/case_7.json")
    assert params.turbulence_model_solver.reconstruction_gradient_limiter == 1.0
    assert params.turbulence_model_solver.model_constants is not None
    assert params.initial_condition is None

    params = fl.Flow360Params("data/cases/case_18.json")
    assert set(params.navier_stokes_solver.low_dissipation_control_factors) == {0.0, 1.0, 2.0, 3.0}

    params = fl.Flow360Params("data/cases/case_13.json")
    assert set(params.surface_output.output_fields) == {"Cp", "Cf", "uhh", "primitiveVars"}
    assert set(params.volume_output.output_fields) == {
        "hmmm",
        "primitiveVars",
        "Mach",
        "SpalartAllmaras_DDES",
    }
    assert set(params.slice_output.slices["mid_Height"].output_fields) == {"uhh", "Cp"}
    assert set(params.iso_surface_output.iso_surfaces["newKey"].output_fields) == {"Mach", "hmmm"}
    assert set(params.monitor_output.monitors["newKey"].output_fields) == {"hmmm"}
    assert set(params.monitor_output.monitors["Group1"].output_fields) == {"primitiveVars", "uhh"}

    params = fl.Flow360Params("data/cases/case_20.json")
    assert set(params.surface_output.output_fields) == {"Cp", "yPlus"}
    assert set(params.slice_output.slices["sliceTest"].output_fields) == {"aggg"}

    # ##:: case_udd_legacy.json has linearSolver instead of linearSovlerConfig(legacy)
    params = fl.Flow360Params("data/cases/case_udd_legacy.json")
    assert params.navier_stokes_solver.linear_solver.max_iterations == 1
    assert params.turbulence_model_solver.linear_solver.max_iterations == 2
    assert params.transition_model_solver.linear_solver.max_iterations == 3

    ##:: case_HeatTransfer.json has linearSolver instead of linearSovlerConfig(legacy)
    params = fl.Flow360Params("data/cases/case_HeatTransfer.json")
    assert params.heat_equation_solver.linear_solver.max_iterations == 4
    assert (
        params.navier_stokes_solver.linear_solver.max_iterations
        == LinearSolverLegacy.__fields__["max_iterations"].default
    )

    ##:: case_18.json has empty heatEquationSolver,
    params = fl.Flow360Params("data/cases/case_18.json")
    assert params.heat_equation_solver is None
    assert params.transition_model_solver is None

    params = fl.Flow360Params("data/cases/case_boundaries.json")
    assert params.boundaries["1"].static_pressure_ratio == 0.4
    assert params.boundaries["2"].Mach == 1.0
    assert params.boundaries["3"].total_pressure_ratio == 1.0
    assert params.boundaries["3"].total_temperature_ratio == 1.0
    assert params.boundaries["4"].mass_flow_rate == 1.0
    assert params.boundaries["5"].mass_flow_rate == 2.0
    assert params.boundaries["5"].total_temperature_ratio == 1.0


def test_version_update():
    files = ["test_version_b16.json"]

    for file in files:
        params = fl.Flow360Params(f"data/cases/{file}")
        assert params


def test_turbulence_updater():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(data_turbulence, temp_file)

    params = fl.Flow360Params(temp_file.name)

    assert params.boundaries["2"].turbulence_quantities.model_type == "ModifiedTurbulentViscosity"

    assert params.freestream.turbulence_quantities.model_type == "ModifiedTurbulentViscosity"


def test_updater_map():
    version_from = "1.2.3"
    version_to = "2.3.4"
    update_map = [
        ("1.2.3", "1.2.4", _no_update),
        ("1.2.4", "2.3.4", _no_update),
    ]

    with pytest.raises(Flow360NotImplementedError):
        update_path = _find_update_path(version_from=version_from, version_to=version_to)

    update_path = _find_update_path(
        version_from=version_from, version_to=version_to, update_map=update_map
    )
    assert len(update_path) == 2

    update_map = [
        ("1.2.*", "2.3.0", _no_update),
        ("2.3.*", "2.3.*", _no_update),
    ]

    update_path = _find_update_path(
        version_from=version_from, version_to=version_to, update_map=update_map
    )
    assert len(update_path) == 2

    update_map = [
        ("1.2.*", "2.3.0", _no_update),
        ("2.3.*", "2.3.3", _no_update),
        ("2.3.3", "2.3.4", _no_update),
    ]
    update_path = _find_update_path(
        version_from=version_from, version_to=version_to, update_map=update_map
    )
    assert len(update_path) == 3

    update_map = [
        ("1.2.*", "2.2.0", _no_update),
        ("2.2.*", "2.3.3", _no_update),
        ("2.3.3", "2.3.4", _no_update),
    ]

    update_path = _find_update_path(
        version_from="1.2.3b17", version_to=version_to, update_map=update_map
    )
    assert len(update_path) == 3

    update_path = _find_update_path(version_from=UPDATE_MAP[0][0], version_to=UPDATE_MAP[-1][1])
    update_path = _find_update_path(version_from=UPDATE_MAP[0][0], version_to=fl.__version__)
