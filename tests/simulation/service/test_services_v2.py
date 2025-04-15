import json
import re

import pytest

from flow360.component.simulation import services
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
)
from tests.utils import compare_dict_to_ref


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
                    "_id": "137854c4-dea1-47a4-b352-b545ffb0b85c",
                }
            ],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": 1.0, "units": "m**2"},
        },
        "time_stepping": {
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "models": [
            {
                "_id": "09435427-c2dd-4535-935c-b131ab7d1a5b",
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "Mysurface",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [],
                            "_id": "Mysurface",
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
                        "_id": "Mysurface",
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
    params_data_from_geo["version"] = "24.11.0"

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
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "user_defined_dynamics": [],
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, root_item_type="Geometry")

    expected_errors = [
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
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
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
            "type_name": "Steady",
            "max_steps": 10,
            "CFL": {"type": "ramp", "initial": 1.5, "final": 1.5, "ramp_steps": 5},
        },
        "user_defined_dynamics": [],
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(params_as_dict=params_data, root_item_type="Geometry")

    expected_errors = [
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
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
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

    ##3: test default values for surface mesh starting point
    data = services.get_default_params(
        unit_system_name="SI", length_unit="cm", root_item_type="SurfaceMesh"
    )
    assert data["reference_geometry"]["area"]["units"] == "cm**2"
    assert data["reference_geometry"]["moment_center"]["units"] == "cm"
    assert data["reference_geometry"]["moment_length"]["units"] == "cm"
    assert data["private_attribute_asset_cache"]["project_length_unit"]["units"] == "cm"
    # to convert tuples to lists:
    data = json.loads(json.dumps(data))
    compare_dict_to_ref(data, "../../ref/simulation/service_init_surface_mesh.json")


def test_validate_init_data_errors():

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="Geometry"
    )
    _, errors, _ = services.validate_model(params_as_dict=data, root_item_type="Geometry")

    expected_errors = [
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

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
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

    expected_errors = [
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

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
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

    expected_errors = [
        {
            "loc": ("operating_condition", "velocity_magnitude"),
            "type": "missing",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
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
        "version": "24.11.5",
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
            "[{'type': 'missing', 'loc': ('meshing', 'surface_max_edge_length'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['SurfaceMesh']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
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
            "[{'type': 'missing', 'loc': ('meshing', 'defaults', 'boundary_layer_first_layer_thickness'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['VolumeMesh']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
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
            "[{'type': 'missing', 'loc': ('operating_condition', 'velocity_magnitude'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['Case']}, 'url': 'https://errors.pydantic.dev/2.7/v/missing'}]"
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


def test_validation_level_intersection():
    def get_validation_levels_to_use(root_item_type, requested_levels):
        avaliable_levels = services._determine_validation_level(
            up_to="Case", root_item_type=root_item_type
        )
        return services._intersect_validation_levels(requested_levels, avaliable_levels)

    assert get_validation_levels_to_use("Geometry", "All") == ["SurfaceMesh", "VolumeMesh", "Case"]

    assert get_validation_levels_to_use("SurfaceMesh", "All") == ["VolumeMesh", "Case"]

    assert get_validation_levels_to_use("VolumeMesh", "All") == [
        "Case",
    ]

    assert get_validation_levels_to_use("SurfaceMesh", ["Case", "VolumeMesh", "SurfaceMesh"]) == [
        "Case",
        "VolumeMesh",
    ]


def validate_proper_unit(obj, allowed_units_string):
    def is_expected_unit(unit_str, allowed_units_string):
        tokens = re.findall(r"[A-Za-z_]+", unit_str)
        return all(token in allowed_units_string for token in tokens)

    if isinstance(obj, dict):
        if "value" in obj and "units" in obj:
            assert is_expected_unit(obj["units"], allowed_units_string)

        for _, val in obj.items():
            validate_proper_unit(val, allowed_units_string)

    elif isinstance(obj, list):
        for item in obj:
            validate_proper_unit(item, allowed_units_string)


def test_imperial_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, new_unit_system="Imperial")
    imperial_units = {"ft", "lb", "s", "degF", "delta_degF", "rad", "degree"}

    validate_proper_unit(dict_to_convert, imperial_units)
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "degF"
    assert abs(temperature_tester["value"] - 302) / 302 < 1e-10

    # General comparison\
    with open("./ref/unit_system_converted_imperial.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(dict_to_convert, ref_dict)


def test_CGS_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, new_unit_system="CGS")
    CGS_units = {"cm", "g", "s", "K", "rad", "degree"}

    validate_proper_unit(dict_to_convert, CGS_units)
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "K"
    assert abs(temperature_tester["value"] - 423.15) / 423.15 < 1e-10

    # General comparison
    with open("./ref/unit_system_converted_CGS.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(dict_to_convert, ref_dict, rtol=1e-7)  # Default tol fail for Windows


def test_SI_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, new_unit_system="SI")
    SI_units = {"m", "kg", "s", "K", "rad", "degree"}

    validate_proper_unit(dict_to_convert, SI_units)
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "K"
    assert abs(temperature_tester["value"] - 423.15) / 423.15 < 1e-10

    # General comparison
    with open("./ref/unit_system_converted_SI.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(dict_to_convert, ref_dict, rtol=1e-7)  # Default tol fail for Windows


def test_updater_service():
    with open("data/updater_should_pass.json", "r") as fp:
        dict_to_update = json.load(fp)
    updated_params_as_dict, errors = services.update_simulation_json(
        params_as_dict=dict_to_update, target_python_api_version="25.2.2"
    )

    with open("ref/updater_to_25_2_2.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(updated_params_as_dict, ref_dict)
    assert not errors

    # ============#
    dict_to_update["version"] = "999.999.999"
    updated_params_as_dict, errors = services.update_simulation_json(
        params_as_dict=dict_to_update, target_python_api_version="25.2.2"
    )
    assert len(errors) == 1
    assert (
        errors[0]
        == "Input `SimulationParams` have higher version than the target version and thus cannot be handled."
    )
