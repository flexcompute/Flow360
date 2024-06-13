import pytest

from flow360.component.simulation import services


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)



def test_init_service():
    data = services.get_default_params("SI", 'm')
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
        "operating_condition": {
            "alpha": {"units": "flow360_angle_unit", "value": 0.5235987755982988},
            "beta": {"units": "flow360_angle_unit", "value": 0.3490658503988659},
            "reference_velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.5},
            "thermal_state": {
                "_cached": {"altitude": 123, "temperature_offset": 1234},
                "density": {"units": "flow360_density_unit", "value": 1.0},
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.37000000000000005,
                        },
                        "reference_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.91,
                        },
                        "reference_viscosity": {
                            "units": "flow360_viscosity_unit",
                            "value": 5.002915451895044e-12,
                        },
                    },
                    "name": "air",
                    "type": "air",
                },
                "temperature": {"units": "flow360_temperature_unit", "value": 1.0},
            },
            "velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.8},
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
        "operating_condition": {
            "alpha": {"units": "flow360_angle_unit", "value": 0.5235987755982988},
            "beta": {"units": "flow360_angle_unit", "value": 0.3490658503988659},
            "reference_velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.5},
            "thermal_state": {
                "_cached": {"altitude": 123, "temperature_offset": 1234},
                "density": {"units": "flow360_density_unit", "value": 1.0},
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.37000000000000005,
                        },
                        "reference_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.91,
                        },
                        "reference_viscosity": {
                            "units": "flow360_viscosity_unit",
                            "value": 5.002915451895044e-12,
                        },
                    },
                    "name": "air",
                    "type": "air",
                },
                "temperature": {"units": "flow360_temperature_unit", "value": 1.0},
            },
            "velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.8},
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
        "operating_condition": {
            "alpha": {"units": "flow360_angle_unit", "value": 0.5235987755982988},
            "beta": {"units": "flow360_angle_unit", "value": 0.3490658503988659},
            "reference_velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.5},
            "thermal_state": {
                "_cached": {"altitude": 123, "temperature_offset": 1234},
                "density": {"units": "flow360_density_unit", "value": 1.0},
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.37000000000000005,
                        },
                        "reference_temperature": {
                            "units": "flow360_temperature_unit",
                            "value": 0.91,
                        },
                        "reference_viscosity": {
                            "units": "flow360_viscosity_unit",
                            "value": 5.002915451895044e-12,
                        },
                    },
                    "name": "air",
                    "type": "air",
                },
                "temperature": {"units": "flow360_temperature_unit", "value": 1.0},
            },
            "velocity_magnitude": {"units": "flow360_velocity_unit", "value": 0.8},
        },
        "user_defined_dynamics": [],
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")

    assert len(errors) == 2
    assert errors[0]["loc"] == ("meshing", "farfield")
    assert errors[1]["loc"] == ("reference_geometry", "area", "value")
