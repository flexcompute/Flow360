import json
import re

import pytest

from flow360.component.simulation import services
from flow360.component.simulation.validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
)
from tests.utils import compare_dict_to_ref, compare_values


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_validate_service():

    params_data_from_vm = {
        "meshing": {
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "defaults": {"surface_edge_growth_rate": 1.5},
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
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "Mysurface",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [],
                        }
                    ]
                },
                "use_wall_function": False,
            }
        ],
        "user_defined_dynamics": [],
        "unit_system": {"name": "SI"},
        "private_attribute_asset_cache": {
            "project_length_unit": None,
            "project_entity_info": {
                "draft_entities": [],
                "type_name": "VolumeMeshEntityInfo",
                "zones": [],
                "boundaries": [
                    {
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_entity_type_name": "Surface",
                        "name": "Mysurface",
                        "private_attribute_full_name": None,
                        "private_attribute_is_interface": False,
                        "private_attribute_tag_key": None,
                        "private_attribute_sub_components": [],
                    }
                ],
            },
        },
    }

    params_data_from_geo = params_data_from_vm
    params_data_from_geo["meshing"]["defaults"] = {
        "surface_edge_growth_rate": 1.5,
        "boundary_layer_first_layer_thickness": "1*m",
        "surface_max_edge_length": "1*m",
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data_from_geo, root_item_type="Geometry"
    )

    assert errors is None

    _, errors, _ = services.validate_model(
        params_as_dict=params_data_from_vm, root_item_type="VolumeMesh", validation_level=CASE
    )

    assert errors is None


def test_validate_error():
    params_data = {
        "meshing": {
            "farfield": "invalid",
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "defaults": {"surface_edge_growth_rate": 1.5},
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
        "unit_system": {"name": "SI"},
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, root_item_type="Geometry")

    excpected_errors = [
        {
            "loc": ("meshing", "defaults", "boundary_layer_first_layer_thickness"),
            "type": "missing",
            "ctx": {"relevant_for": ["VolumeMesh"]},
        },
        {
            "loc": ("meshing", "defaults", "surface_max_edge_length"),
            "type": "missing",
            "ctx": {"relevant_for": ["SurfaceMesh"]},
        },
        {
            "loc": ("meshing", "farfield"),
            "type": "extra_forbidden",
            "ctx": {"relevant_for": ["SurfaceMesh", "VolumeMesh"]},
        },
    ]
    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_validate_multiple_errors():
    params_data = {
        "meshing": {
            "farfield": "invalid",
            "refinement_factor": 1.0,
            "gap_treatment_strength": 0.2,
            "defaults": {
                "surface_edge_growth_rate": 1.5,
                "boundary_layer_first_layer_thickness": "1*m",
                "surface_max_edge_length": "1*s",
            },
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
        "unit_system": {"name": "SI"},
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, root_item_type="Geometry")

    excpected_errors = [
        {
            "loc": ("meshing", "defaults", "surface_max_edge_length"),
            "type": "value_error",
            "ctx": {"relevant_for": ["SurfaceMesh"]},
        },
        {
            "loc": ("meshing", "farfield"),
            "type": "extra_forbidden",
            "ctx": {"relevant_for": ["SurfaceMesh", "VolumeMesh"]},
        },
        {
            "loc": ("reference_geometry", "area", "value"),
            "type": "greater_than",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]
    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


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
            "defaults": {"surface_edge_growth_rate": 1.2},
        },
        "unit_system": {"name": "SI"},
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, root_item_type="Geometry")
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
    _, errors, _ = services.validate_model(params_as_dict=data, root_item_type="Geometry")

    excpected_errors = [
        {
            "loc": ("meshing", "defaults", "boundary_layer_first_layer_thickness"),
            "type": "missing",
            "ctx": {"relevant_for": ["VolumeMesh"]},
        },
        {
            "loc": ("meshing", "defaults", "surface_max_edge_length"),
            "type": "missing",
            "ctx": {"relevant_for": ["SurfaceMesh"]},
        },
        {
            "loc": ("operating_condition", "velocity_magnitude"),
            "type": "missing",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_validate_init_data_for_sm_and_vm_errors():

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="Geometry"
    )
    _, errors, _ = services.validate_model(
        params_as_dict=data,
        root_item_type="Geometry",
        validation_level=[SURFACE_MESH, VOLUME_MESH],
    )

    excpected_errors = [
        {
            "loc": ("meshing", "defaults", "boundary_layer_first_layer_thickness"),
            "type": "missing",
            "ctx": {"relevant_for": ["VolumeMesh"]},
        },
        {
            "loc": ("meshing", "defaults", "surface_max_edge_length"),
            "type": "missing",
            "ctx": {"relevant_for": ["SurfaceMesh"]},
        },
    ]

    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_validate_init_data_vm_workflow_errors():

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="VolumeMesh"
    )
    _, errors, _ = services.validate_model(
        params_as_dict=data, root_item_type="VolumeMesh", validation_level=CASE
    )

    excpected_errors = [
        {
            "loc": ("operating_condition", "velocity_magnitude"),
            "type": "missing",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_front_end_JSON_with_multi_constructor():
    params_data = {
        "meshing": {
            "defaults": {
                "boundary_layer_first_layer_thickness": "1*m",
                "surface_max_edge_length": "1*m",
            },
            "refinement_factor": 1.45,
            "refinements": [
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_entity_type_name": "Box",
                                "private_attribute_id": "hardcoded_id-1",
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
                                "private_attribute_id": "hardcoded_id-2",
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
                                "private_attribute_id": "hardcoded_id-3",
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
                        "private_attribute_id": "hardcoded_id-4",
                        "name": "automated_farfied_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
        "private_attribute_asset_cache": {
            "project_length_unit": "m",
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x_1", "face_x_2", "face_x_3"],
                "face_group_tag": "some_tag",
                "face_attribute_names": ["some_tag"],
                "grouped_faces": [
                    [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                ],
            },
        },
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                },
                "use_wall_function": False,
            }
        ],
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
        params_as_dict=params_data, root_item_type="Geometry"
    )
    assert errors is None
    with open("../../ref/simulation/simulation_json_with_multi_constructor_used.json", "r") as f:
        ref_data = json.load(f)
        ref_param, err, _ = services.validate_model(
            params_as_dict=ref_data, root_item_type="Geometry"
        )
        assert err is None

    param_dict = simulation_param.model_dump(exclude_none=True)
    ref_param_dict = ref_param.model_dump(exclude_none=True)
    assert compare_values(ref_param_dict, param_dict)


def test_generate_process_json():
    params_data = {
        "meshing": {
            "defaults": {
                # "boundary_layer_first_layer_thickness": "1*m",
                # "surface_max_edge_length": "1*m",
            },
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
            "private_attribute_constructor": "default",
            "private_attribute_input_cache": {},
            "alpha": {"value": 5.0, "units": "degree"},
            "beta": {"value": 0.0, "units": "degree"},
            # "velocity_magnitude": {"value": 0.8, "units": "km/s"},
        },
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                },
                "use_wall_function": False,
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": "m",
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x_1", "face_x_2", "face_x_3"],
                "face_group_tag": "some_tag",
                "face_attribute_names": ["some_tag"],
                "grouped_faces": [
                    [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                ],
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            "[Internal] Validation error occurred for supposedly validated param! Errors are: [{'type': 'missing', 'loc': ('meshing', 'surface_max_edge_length'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['SurfaceMesh']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
        ),
    ):
        res1, res2, res3 = services.generate_process_json(
            simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="SurfaceMesh"
        )

    params_data["meshing"]["defaults"]["surface_max_edge_length"] = "1*m"
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="SurfaceMesh"
    )

    assert res1 is not None
    assert res2 is None
    assert res3 is None

    with pytest.raises(
        ValueError,
        match=re.escape(
            "[Internal] Validation error occurred for supposedly validated param! Errors are: [{'type': 'missing', 'loc': ('meshing', 'defaults', 'boundary_layer_first_layer_thickness'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['VolumeMesh']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
        ),
    ):
        res1, res2, res3 = services.generate_process_json(
            simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="VolumeMesh"
        )

    params_data["meshing"]["defaults"]["boundary_layer_first_layer_thickness"] = "1*m"
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="VolumeMesh"
    )

    assert res1 is not None
    assert res2 is not None
    assert res3 is None

    with pytest.raises(
        ValueError,
        match=re.escape(
            "[Internal] Validation error occurred for supposedly validated param! Errors are: [{'type': 'missing', 'loc': ('operating_condition', 'velocity_magnitude'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['Case']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
        ),
    ):
        res1, res2, res3 = services.generate_process_json(
            simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="Case"
        )

    params_data["operating_condition"]["velocity_magnitude"] = {"value": 0.8, "units": "km/s"}
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="Case"
    )

    assert res1 is not None
    assert res2 is not None
    assert res3 is not None
