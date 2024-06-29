import json

import pytest

from flow360.component.simulation import services


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_init_service():
    data = services.get_default_params("SI", "m")
    print(data)
    assert data


def test_validate_service():
    params_data = {
        "meshing": {
            "farfield": "auto",
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "surface_layer_growth_rate": 1.5,
            "refinements": [],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": 1.0, "units": "m**2"},
        },
        "time_stepping": {
            "order_of_accuracy": 2,
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "user_defined_dynamics": [],
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")

    assert errors is None


def test_validate_error():
    params_data = {
        "meshing": {
            "farfield": "invalid",
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "surface_layer_growth_rate": 1.5,
            "refinements": [],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": 1.0, "units": "m**2"},
        },
        "time_stepping": {
            "order_of_accuracy": 2,
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "user_defined_dynamics": [],
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")

    assert len(errors) == 1
    assert errors[0]["loc"] == ("meshing", "farfield")


def test_validate_multiple_errors():
    params_data = {
        "meshing": {
            "farfield": "invalid",
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "surface_layer_growth_rate": 1.5,
            "refinements": [],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": -10.0, "units": "m**2"},
        },
        "time_stepping": {
            "order_of_accuracy": 2,
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "user_defined_dynamics": [],
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")

    assert len(errors) == 2
    assert errors[0]["loc"] == ("meshing", "farfield")
    assert errors[1]["loc"] == ("reference_geometry", "area", "value")


def test_validate_errors():

    params_data = {
        "meshing": {
            "farfield": "auto",
            "refinement_factor": 1,
            "refinements": [
                {
                    "_id": "926a6cbd-0ddb-4051-b860-3414565e6408",
                    "curvature_resolution_angle": 10,
                    "max_edge_length": 0.1,
                    "name": "Surface refinement_0",
                    "refinement_type": "SurfaceRefinement",
                },
                {
                    "_id": "3972fbbf-4af6-4ca5-a8bb-341bbcf1294b",
                    "first_layer_thickness": 0.001,
                    "name": "Boundary layer refinement_1",
                    "refinement_type": "BoundaryLayer",
                },
            ],
            "surface_layer_growth_rate": 1.2,
            "volume_zones": [],
        },
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")
    json.dumps(errors)


def test_init():

    data = services.get_default_params(unit_system_name="SI", length_unit="m")
    assert data["meshing"]["farfield"] == "auto"
    assert data["operating_condition"]["alpha"]["value"] == 0
    assert data["operating_condition"]["alpha"]["units"] == "degree"
    assert "velocity_magnitude" not in data["operating_condition"].keys()


def test_validate_init_data_errors():

    data = services.get_default_params(unit_system_name="SI", length_unit="m")
    _, errors, _ = services.validate_model(params_as_dict=data, unit_system_name="SI")
    json.dumps(errors)
    assert len(errors) == 1


def test_validate():
    params_str = '{"version":"24.3.0","unit_system":{"name":"SI"},"private_attribute_asset_cache":{"asset_entity_registry":{"internal_registry":{}}},"meshing":{"farfield":"auto","refinement_factor":1,"surface_layer_growth_rate":1.2,"refinements":[{"name":"Surface edge refinement 1","refinement_type":"SurfaceEdgeRefinement","_id":"1e36b5cf-23c6-499d-b2e1-12a2474853a3","type":"projectAnisoSpacing","method":"aspectRatio","value":-4,"edges":{"value":["iGf-HfizH8LFOFqa","8elVtrQVeW9CsFMD"]}},{"name":"Axisymmetric refinement 2","refinement_type":"AxisymmetricRefinement","_id":"153c5329-7d6c-40d6-9402-b02180afa41f"}],"volume_zones":[],"gap_treatment_strength":1},"operating_condition":{"type_name":"AerospaceCondition","private_attribute_constructor":"default","private_attribute_input_cache":{},"alpha":{"value":0,"units":"rad"},"beta":{"value":0,"units":"rad"},"thermal_state":{"type_name":"ThermalState","private_attribute_constructor":"default","private_attribute_input_cache":{},"temperature":{"value":288.15,"units":"K"},"density":{"value":1.225,"units":"kg/m**3"},"material":{"type":"air","name":"air","dynamic_viscosity":{"reference_viscosity":{"value":0.00001716,"units":"Pa*s"},"reference_temperature":{"value":273.15,"units":"K"},"effective_temperature":{"value":110.4,"units":"K"}}}},"velocity_magnitude":{"value":100,"units":"m/s"}},"time_stepping":{"order_of_accuracy":2,"type_name":"Steady","max_steps":700,"CFL":{"type":"adaptive","min":0.1,"max":10000,"max_relative_change":1,"convergence_limiting_factor":0.25}},"user_defined_dynamics":null,"outputs":null}'
    _, errors, _ = services.validate_model(
        params_as_dict=json.loads(params_str), unit_system_name="SI"
    )
    json.dumps(errors)
    assert len(errors) == 1
