import json

import pytest

from flow360.component.simulation import services
from tests.utils import compare_dict_to_ref, compare_values


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_validate_service():

    params_data = {
        "meshing": {
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "surface_layer_growth_rate": 1.5,
            "refinements": [],
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
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
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
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
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
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
            "curvature_resolution_angle": 10,
            "refinements": [
                {
                    "_id": "926a6cbd-0ddb-4051-b860-3414565e6408",
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
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
            "surface_layer_growth_rate": 1.2,
        },
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, unit_system_name="SI")
    json.dumps(errors)


def test_init():
    ##1: test default values for geometry starting point
    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="Geometry"
    )
    assert data["operating_condition"]["alpha"]["value"] == 0
    assert data["operating_condition"]["alpha"]["units"] == "degree"
    assert "velocity_magnitude" not in data["operating_condition"].keys()
    # to convert tuples to lists:
    data = json.loads(json.dumps(data))
    compare_dict_to_ref(data, "../../ref/simulation/service_init_geometry.json")

    ##2: test default values for volume mesh starting point
    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="VolumeMesh"
    )
    assert "meshing" not in data
    # to convert tuples to lists:
    data = json.loads(json.dumps(data))
    compare_dict_to_ref(data, "../../ref/simulation/service_init_volume_mesh.json")


def test_validate_init_data_errors():

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="Geometry"
    )
    _, errors, _ = services.validate_model(params_as_dict=data, unit_system_name="SI")
    assert len(errors) == 3
    assert errors[0]["loc"][-1] == "max_edge_length"
    assert errors[0]["type"] == "missing"
    assert errors[1]["loc"][-1] == "first_layer_thickness"
    assert errors[1]["type"] == "missing"
    assert errors[2]["loc"][-1] == "velocity_magnitude"
    assert errors[2]["type"] == "missing"

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="VolumeMesh"
    )
    _, errors, _ = services.validate_model(params_as_dict=data, unit_system_name="SI")
    assert len(errors) == 1
    assert errors[0]["loc"][-1] == "velocity_magnitude"
    assert errors[0]["type"] == "missing"


def test_front_end_JSON_with_multi_constructor():
    params_data = {
        "meshing": {
            "refinement_factor": 1.45,
            "refinements": [
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_entity_type_name": "Box",
                                "name": "my_box_default",
                                "private_attribute_zone_boundary_names": {"items": []},
                                "type_name": "Box",
                                "private_attribute_constructor": "default",
                                "private_attribute_input_cache": {},
                                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                                "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                                "axis_of_rotation": [1.0, 0.0, 0.0],
                                "angle_of_rotation": {"value": 20.0, "units": "degree"},
                            },
                            {
                                "type_name": "Box",
                                "private_attribute_constructor": "from_principal_axes",
                                "private_attribute_input_cache": {
                                    "axes": [[0.6, 0.8, 0.0], [0.8, -0.6, 0.0]],
                                    "center": {"value": [7.0, 1.0, 2.0], "units": "m"},
                                    "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                                    "name": "my_box_from",
                                },
                            },
                            {
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_entity_type_name": "Cylinder",
                                "name": "my_cylinder_default",
                                "private_attribute_zone_boundary_names": {"items": []},
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                                "height": {"value": 3.0, "units": "m"},
                                "outer_radius": {"value": 2.0, "units": "m"},
                            },
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "cm", "value": 7.5},
                }
            ],
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "private_attribute_constructor": "from_mach",
            "private_attribute_input_cache": {
                "alpha": {"value": 5.0, "units": "degree"},
                "beta": {"value": 0.0, "units": "degree"},
                "thermal_state": {
                    "type_name": "ThermalState",
                    "private_attribute_constructor": "from_standard_atmosphere",
                    "private_attribute_input_cache": {
                        "altitude": {"value": 1000.0, "units": "m"},
                        "temperature_offset": {"value": 0.0, "units": "K"},
                    },
                },
                "mach": 0.8,
            },
        },
    }

    simulation_param, errors, _ = services.validate_model(
        params_as_dict=params_data, unit_system_name="SI"
    )
    assert errors is None
    with open("../../ref/simulation/simulation_json_with_multi_constructor_used.json", "r") as f:
        ref_data = json.load(f)
    param_dict = simulation_param.model_dump(exclude_none=True)
    compare_values(ref_data, param_dict)
