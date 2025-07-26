import json
import re

import pytest
from unyt import Unit

import flow360.component.simulation.units as u
from flow360.component.simulation import services
from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.unit_system import _PredefinedUnitSystem
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
    get_validation_info,
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
                        "name": "automated_farfield_entity",
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

    params_data_op_from_mach_reynolds = params_data_from_vm.copy()
    params_data_op_from_mach_reynolds["private_attribute_asset_cache"]["project_length_unit"] = {
        "value": 0.8059,
        "units": "m",
    }
    params_data_op_from_mach_reynolds["operating_condition"] = {
        "type_name": "AerospaceCondition",
        "private_attribute_constructor": "from_mach_reynolds",
        "private_attribute_input_cache": {
            "mach": 0.84,
            "reynolds_mesh_unit": 10.0,
            "alpha": {"value": 3.06, "units": "degree"},
            "beta": {"value": 0.0, "units": "degree"},
            "temperature": {"value": 288.15, "units": "K"},
        },
        "alpha": {"value": 3.06, "units": "degree"},
        "beta": {"value": 0.0, "units": "degree"},
        "velocity_magnitude": {
            "type_name": "number",
            "value": 285.84696487889875,
            "units": "m/s",
        },
        "thermal_state": {
            "type_name": "ThermalState",
            "private_attribute_constructor": "default",
            "private_attribute_input_cache": {},
            "temperature": {"value": 288.15, "units": "K"},
            "density": {"value": 7.767260032496146e-07, "units": "Pa*s**2/m**2"},
            "material": {
                "type": "air",
                "name": "air",
                "dynamic_viscosity": {
                    "reference_viscosity": {"value": 1.716e-05, "units": "Pa*s"},
                    "reference_temperature": {"value": 273.15, "units": "K"},
                    "effective_temperature": {"value": 110.4, "units": "K"},
                },
            },
        },
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data_from_geo,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )

    assert errors is None

    _, errors, _ = services.validate_model(
        params_as_dict=params_data_from_vm,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level=CASE,
    )

    assert errors is None

    _, errors, _ = services.validate_model(
        params_as_dict=params_data_op_from_mach_reynolds,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level=CASE,
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
                        "name": "automated_farfield_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1.0, "units": "m"},
            "area": {"value": 1.0, "units": "m**2", "type_name": "number"},
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

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
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
                        "name": "automated_farfield_entity",
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

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )

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
                        "name": "automated_farfield_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
            "defaults": {"surface_edge_growth_rate": 1.2},
        },
        "unit_system": {"name": "SI"},
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    json.dumps(errors)


def test_validate_error_from_initialize_variable_space():
    with open("../translator/data/simulation_isosurface.json", "r") as fp:
        param_dict = json.load(fp)

    a = UserVariable(name="my_time_stepping_var", value=0.6 * u.s)
    _, errors, _ = services.validate_model(
        params_as_dict=param_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": ["unknown"],
            "msg": "Loading user variable 'my_time_stepping_var' from simulation.json "
            "which is already defined in local context. Please change your local user variable definition.",
        }
    ]
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["msg"] == exp_err["msg"]

    services.clear_context()
    _ = UserVariable(name="my_time_stepping_var", value=0.6 * u.s)
    _, errors, _ = services.validate_model(
        params_as_dict=param_dict,
        validated_by=services.ValidationCalledBy.SERVICE,
        root_item_type="VolumeMesh",
    )
    assert errors is None


def test_validate_error_from_multi_constructor():

    def _compare_validation_errors(err, exp_err):
        assert len(errors) == len(expected_errors)
        for err, exp_err in zip(errors, expected_errors):
            for key in exp_err.keys():
                assert err[key] == exp_err[key]

    # test from_mach() with two validation errors within private_attribute_input_cache
    params_data = {
        "operating_condition": {
            "private_attribute_constructor": "from_mach",
            "type_name": "AerospaceCondition",
            "private_attribute_input_cache": {
                "mach": -1,
                "alpha": {"value": 0, "units": "degree"},
                "beta": {"value": 0, "units": "degree"},
                "thermal_state": {
                    "type_name": "ThermalState",
                    "private_attribute_constructor": "default",
                    "density": {"value": -2, "units": "kg/m**3"},
                    "temperature": {"value": 288.15, "units": "K"},
                },
            },
        },
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    expected_errors = [
        {
            "loc": ("operating_condition", "private_attribute_input_cache", "mach"),
            "type": "greater_than_equal",
            "msg": "Input should be greater than or equal to 0",
            "input": -1,
            "ctx": {"ge": "0.0"},
        },
        {
            "loc": (
                "operating_condition",
                "private_attribute_input_cache",
                "thermal_state",
                "density",
                "value",
            ),
            "type": "greater_than",
            "msg": "Input should be greater than 0",
            "input": -2,
            "ctx": {"gt": "0.0"},
        },
    ]
    _compare_validation_errors(errors, expected_errors)

    # test BETDisk.from_dfdc() with:
    #   1. one validation error within private_attribute_input_cache
    #   2. a missing BETDiskCache argument for the dfdc constructor
    #   3. one validation error outside the input_cache
    params_data = {
        "models": [
            {
                "name": "BET disk",
                "private_attribute_constructor": "from_dfdc",
                "private_attribute_input_cache": {
                    "angle_unit": {"units": "degree", "value": 1.0},
                    "blade_line_chord": {"units": "m", "value": 5.0},
                    "chord_ref": {"units": "m", "value": -14.0},
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [0.0, 0.0, 1.0],
                                "center": {"units": "m", "value": [0.0, 0.0, 0.0]},
                                "height": {"units": "m", "value": -15.0},
                                "inner_radius": {"units": "m", "value": 0.0},
                                "name": "BET_cylinder",
                                "outer_radius": {"units": "m", "value": 3.81},
                                "private_attribute_entity_type_name": "Cylinder",
                                "private_attribute_full_name": None,
                                "private_attribute_id": "ca0d3a3f-49cb-4637-a789-744c643ce955",
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_zone_boundary_names": {"items": []},
                            }
                        ]
                    },
                    "file": {
                        "content": "DFDC Version 0.70E+03\nvb block 1 c 25 HP SLS\n\nOPER\n!   Vinf         Vref         RPM1\n   0.000       10.000         15.0\n!   Rho          Vso          Rmu          Alt\n   1.0          342.0      0.17791E-04  0.30000E-01\n!  XDwake             Nwake\n   1.0000               20\n!        Lwkrlx\n            F\nENDOPER\n\nAERO\n!  #sections\n     5\n!  Xisection\n  0.09\n  !       A0deg        dCLdA        CLmax         CLmin\n    -6.5           4.000       1.2500     -0.0000\n  !  dCLdAstall     dCLstall      Cmconst         Mcrit\n    0.10000      0.10000     -0.10000      0.75000\n  !       CDmin      CLCDmin     dCDdCL^2\n    0.075000E-01  0.00000      0.40000E-02\n  !       REref        REexp\n    0.30000E+06 -0.70000\n    0.17\n  !       A0deg        dCLdA        CLmax         CLmin\n          -6.0         6.0       1.300     -0.55000\n  !  dCLdAstall     dCLstall      Cmconst         Mcrit\n      0.10000      0.10000     -0.10000      0.75000\n  !       CDmin      CLCDmin     dCDdCL^2\n      0.075000E-01  0.10000      0.40000E-02\n  !       REref        REexp\n      0.30000E+06 -0.70000\n     0.51\n  !       A0deg        dCLdA        CLmax         CLmin\n     -1.0       6.00                1.400     -1.4000\n  !  dCLdAstall     dCLstall      Cmconst         Mcrit\n       0.10000      0.10000     -0.10000      0.75000\n  !       CDmin      CLCDmin     dCDdCL^2\n       0.05000E-01  0.10000      0.40000E-02\n  !       REref        REexp\n       0.30000E+06 -0.70000\n    0.8\n  !       A0deg        dCLdA        CLmax         CLmin\n         -1.0      6.0       1.600     -1.500\n  !  dCLdAstall     dCLstall      Cmconst         Mcrit\n           0.10000      0.10000     -0.10000      0.75000\n  !       CDmin      CLCDmin     dCDdCL^2\n           0.03000E-01  0.10000      0.40000E-02\n  !       REref        REexp\n           0.30000E+06 -0.70000\n    1.0\n  !       A0deg        dCLdA        CLmax         CLmin\n            -1          6.0           1.0     -1.8000\n  !  dCLdAstall     dCLstall      Cmconst         Mcrit\n        0.10000      0.10000     -0.10000      0.75000\n  !       CDmin      CLCDmin     dCDdCL^2\n          0.04000E-01  0.10000      0.40000E-02\n  !       REref        REexp\n        0.30000E+06 -0.70000\nENDAERO\n\nROTOR\n!  Xdisk               Nblds       NRsta\n  150                   3           63\n!  #stations\n    63\n!      r        C       Beta0deg\n0.087645023\t0.432162215\t33.27048712\n0.149032825\t0.432162215\t32.37853609\n0.230063394\t0.432162215\t31.42712165\n0.31845882\t0.432162215\t30.65409742\n0.354421836\t0.432162215\t30.13214089\n0.412445831\t0.432162215\t29.41523066\n0.465733177\t0.425230275\t28.75567325\n0.535598785\t0.418298338\t27.89538097\n0.607834576\t0.411366397\t26.69097178\n0.657570537\t0.404434457\t25.88803232\n0.695464717\t0.39750252\t25.25715132\n0.733359236\t0.390570579\t24.5689175\n0.773621567\t0.383638638\t23.93803649\n0.818620882\t0.376706702\t23.19244985\n0.86598886\t0.369774761\t22.36083398\n0.912171916\t0.36284282\t21.67260016\n0.958355818\t0.355910895\t20.84098429\n1.008092457\t0.355906578\t19.92333919\n1.051908539\t0.355902261\t19.03437051\n1.090987133\t0.355897941\t18.34613668\n1.134803219\t0.355893624\t17.457168\n1.17980135\t0.355889307\t16.91231622\n1.229536976\t0.355884987\t16.16672958\n1.275719693\t0.35588067\t15.53584858\n1.314797781\t0.355876353\t14.93364398\n1.358613021\t0.355872033\t14.18805734\n1.398875352\t0.355867716\t13.55717634\n1.42966541\t0.355863399\t12.86894251\n1.468744004\t0.355859079\t12.18070869\n1.51019075\t0.355854762\t11.49247487\n1.549268331\t0.355850445\t10.9762995\n1.596633604\t0.355846125\t10.60350618\n1.6629458\t0.355841808\t9.94394877\n1.717417226\t0.355837491\t9.28439136\n1.773072894\t0.355833171\t8.59615753\n1.81688746\t0.355828854\t7.96527653\n1.866622407\t0.355824537\t7.33439553\n1.893858878\t0.355820217\t6.87557298\n1.949512352\t0.3558159\t6.56013248\n2.015823535\t0.355811583\t6.07263352\n2.07503076\t0.355807263\t5.49910533\n2.124764355\t0.355802946\t5.0976356\n2.169761643\t0.355798629\t4.69616587\n2.231337023\t0.355794309\t4.12263769\n2.273965822\t0.355789992\t3.77852078\n2.321330759\t0.355785675\t3.46308028\n2.395930475\t0.355781355\t2.97558132\n2.533289143\t0.355777038\t2.0005834\n2.61499265\t0.355772721\t1.62779008\n2.664726078\t0.355768401\t1.25499676\n2.709722688\t0.355764084\t0.96823267\n2.768929239\t0.355759767\t0.50941012\n2.839977235\t0.355755447\t-0.064118065\n2.906288246\t0.35575113\t-0.522940614\n3.03772619\t0.355746813\t-1.44058571\n3.208240195\t0.355742493\t-2.61631849\n3.342045113\t0.355738176\t-3.333228722\n3.481771091\t0.355733859\t-4.164844591\n3.56702767\t0.355729539\t-4.681019957\n3.647547098\t0.355725222\t-5.053813278\n3.725699048\t0.355720905\t-5.541312235\n3.770695491\t0.355716585\t-5.799399919\n3.81\t0.355714554\t-6.1\nENDROTOR\n\nGEOM\n    /IGNORED BELOW THIS POINT\n",
                        "file_path": "aba",
                        "type_name": "DFDCFile",
                    },
                    "initial_blade_direction": [1.0, 0.0, 0.0],
                    "length_unit": {"units": "m", "value": 1.0},
                    "omega": {"units": "degree/s", "value": 0.0046},
                    "rotation_direction_rule": "leftHand",
                    "number_of_blades": 2,
                },
                "type": "BETDisk",
                "type_name": "BETDisk",
            },
            {
                "entities": {"stored_entities": []},
                "heat_spec": {"type_name": "HeatFlux", "value": {"units": "W/m**2", "value": 0.0}},
                "name": "Wall",
                "roughness_height": {"units": "m", "value": -10.0},
                "type": "Wall",
                "use_wall_function": False,
                "velocity": None,
            },
        ],
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    expected_errors = [
        {
            "loc": ("models", 0, "private_attribute_input_cache", "chord_ref", "value"),
            "type": "greater_than",
            "msg": "Input should be greater than 0",
            "input": -14,
            "ctx": {"gt": "0.0"},
        },
        {
            "type": "missing_argument",
            "loc": ("models", 0, "private_attribute_input_cache", "n_loading_nodes"),
            "msg": "Missing required argument",
            "ctx": {},
        },
        {
            "loc": (
                "models",
                0,
                "private_attribute_input_cache",
                "entities",
                "stored_entities",
                0,
                "height",
                "value",
            ),
            "type": "greater_than",
            "msg": "Input should be greater than 0",
            "input": -15,
            "ctx": {"gt": "0.0"},
        },
    ]

    _compare_validation_errors(errors, expected_errors)

    # test Box.from_principal_axes() with one validation error within private_attribute_input_cache
    # the multiconstructor call is within a default constructor call
    params_data = {
        "models": [
            {
                "darcy_coefficient": {"units": "m**(-2)", "value": [1000000.0, 0.0, 0.0]},
                "entities": {
                    "stored_entities": [
                        {
                            "angle_of_rotation": {"units": "rad", "value": -2.0943951023931953},
                            "axis_of_rotation": [
                                -0.5773502691896257,
                                -0.5773502691896257,
                                -0.5773502691896261,
                            ],
                            "center": {"units": "m", "value": [0, 0, 0]},
                            "name": "porous_zone",
                            "private_attribute_constructor": "from_principal_axes",
                            "private_attribute_entity_type_name": "Box",
                            "private_attribute_full_name": None,
                            "private_attribute_id": "69751367-210b-4df3-b4cd-1f2adbd866ed",
                            "private_attribute_input_cache": {
                                "axes": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                "center": {"units": "m", "value": [0, 0, 0]},
                                "name": "porous_zone",
                                "size": {"units": "m", "value": [0.2, 0.3, -2.0]},
                            },
                            "private_attribute_registry_bucket_name": "VolumetricEntityType",
                            "private_attribute_zone_boundary_names": {"items": []},
                            "size": {"units": "m", "value": [0.2, 0.3, 2.0]},
                            "type_name": "Box",
                        }
                    ]
                },
                "forchheimer_coefficient": {"units": "1/m", "value": [1, 0, 0]},
                "name": "Porous medium",
                "type": "PorousMedium",
                "volumetric_heat_source": {"units": "W/m**3", "value": 1.0},
            }
        ],
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": (
                "models",
                0,
                "entities",
                "stored_entities",
                0,
                "private_attribute_input_cache",
                "size",
            ),
            "msg": "Value error, arg '[ 0.2  0.3 -2. ] m' cannot have negative value",
            "input": {"units": "m", "value": [0.2, 0.3, -2.0]},
            "ctx": {"error": "arg '[ 0.2  0.3 -2. ] m' cannot have negative value"},
        }
    ]
    _compare_validation_errors(errors, expected_errors)

    # test ThermalState.from_standard_atmosphere() with one validation error within private_attribute_input_cache
    # the multiconstructor call is nested in another multiconstructor call
    params_data = {
        "operating_condition": {
            "alpha": {"units": "degree", "value": 0.0},
            "beta": {"units": "degree", "value": 0.0},
            "private_attribute_constructor": "from_mach",
            "private_attribute_input_cache": {
                "alpha": {"units": "degree", "value": 0.0},
                "beta": {"units": "degree", "value": 0.0},
                "mach": -1,
                "reference_mach": None,
                "thermal_state": {
                    "density": {"units": "kg/m**3", "value": 1.1724995324950298},
                    "material": {
                        "dynamic_viscosity": {
                            "effective_temperature": {"units": "K", "value": 110.4},
                            "reference_temperature": {"units": "K", "value": 273.15},
                            "reference_viscosity": {"units": "Pa*s", "value": 1.716e-05},
                        },
                        "name": "air",
                        "type": "air",
                    },
                    "private_attribute_constructor": "from_standard_atmosphere",
                    "private_attribute_input_cache": {
                        "altitude": {"units": "m", "value": 100.0},
                        "temperature_offset": {"units": "K", "value": 10.0},
                    },
                    "temperature": {"units": "K", "value": 297.5000102251644},
                    "type_name": "ThermalState",
                },
            },
            "reference_velocity_magnitude": None,
            "thermal_state": {
                "density": {"units": "kg/m**3", "value": 1.1724995324950298},
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {"units": "K", "value": 110.4},
                        "reference_temperature": {"units": "K", "value": 273.15},
                        "reference_viscosity": {"units": "Pa*s", "value": 1.716e-05},
                    },
                    "name": "air",
                    "type": "air",
                },
                "private_attribute_constructor": "from_standard_atmosphere",
                "private_attribute_input_cache": {
                    "altitude": {"units": "K", "value": 100.0},
                    "temperature_offset": {"units": "K", "value": 10.0},
                },
                "temperature": {"units": "K", "value": 297.5000102251644},
                "type_name": "ThermalState",
            },
            "type_name": "AerospaceCondition",
            "velocity_magnitude": {"units": "m/s", "value": 34.57709313392731},
        },
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
    }

    _, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    expected_errors = [
        {
            "type": "value_error",
            "loc": (
                "operating_condition",
                "thermal_state",
                "private_attribute_input_cache",
                "altitude",
            ),
            "msg": "Value error, arg '100.0 K' does not match (length) dimension.",
            "input": None,
            "ctx": {"error": "arg '100.0 K' does not match (length) dimension."},
        }
    ]
    _compare_validation_errors(errors, expected_errors)


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

    assert data["models"][0]["roughness_height"]["units"] == "cm"
    # to convert tuples to lists:
    data = json.loads(json.dumps(data))
    compare_dict_to_ref(data, "../../ref/simulation/service_init_surface_mesh.json")


def test_validate_init_data_errors():

    data = services.get_default_params(
        unit_system_name="SI", length_unit="m", root_item_type="Geometry"
    )
    _, errors, _ = services.validate_model(
        params_as_dict=data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
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
        validated_by=services.ValidationCalledBy.LOCAL,
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
        params_as_dict=data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level=CASE,
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
                        "name": "automated_farfield_entity",
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
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    assert errors is None
    with open("../../ref/simulation/simulation_json_with_multi_constructor_used.json", "r") as f:
        ref_data = json.load(f)
        ref_param, err, _ = services.validate_model(
            params_as_dict=ref_data,
            root_item_type="Geometry",
            validated_by=services.ValidationCalledBy.LOCAL,
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
                        "name": "automated_farfield_entity",
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
            "[{'type': 'missing', 'loc': ('meshing', 'surface_max_edge_length'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['SurfaceMesh']}, 'url': 'https://errors.pydantic.dev/2.11/v/missing'}]"
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
            "[{'type': 'missing', 'loc': ('meshing', 'defaults', 'boundary_layer_first_layer_thickness'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['VolumeMesh']}, 'url': 'https://errors.pydantic.dev/2.11/v/missing'}]"
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
            "[{'type': 'missing', 'loc': ('operating_condition', 'velocity_magnitude'), 'msg': 'Field required', 'input': None, 'ctx': {'relevant_for': ['Case']}, 'url': 'https://errors.pydantic.dev/2.11/v/missing'}]"
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


def test_default_validation_contest():
    "Ensure that the default validation context is None which is the escaper for many validators"
    assert get_validation_info() is None


def test_validation_level_intersection():
    def get_validation_levels_to_use(root_item_type, requested_levels):
        available_levels = services._determine_validation_level(
            up_to="Case", root_item_type=root_item_type
        )
        return services._intersect_validation_levels(requested_levels, available_levels)

    assert get_validation_levels_to_use("Geometry", "All") == ["SurfaceMesh", "VolumeMesh", "Case"]

    assert get_validation_levels_to_use("SurfaceMesh", "All") == ["VolumeMesh", "Case"]

    assert get_validation_levels_to_use("VolumeMesh", "All") == [
        "Case",
    ]

    assert get_validation_levels_to_use("SurfaceMesh", ["Case", "VolumeMesh", "SurfaceMesh"]) == [
        "Case",
        "VolumeMesh",
    ]


def test_forward_compatibility_error():

    from flow360.version import __version__

    # Mock a future simulation.json
    with open("data/updater_should_pass.json", "r") as fp:
        future_dict = json.load(fp)
    future_dict["version"] = "99.99.99"
    _, errors, _ = services.validate_model(
        params_as_dict=future_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )

    assert errors[0] == {
<<<<<<< HEAD
        "type": "99.99.99 > 25.6.2b2",
=======
        "type": f"99.99.99 > {__version__}",
>>>>>>> 8d0cbbdd (Added version command and better project loading error (#1295))
        "loc": [],
        "msg": f"The cloud `SimulationParam` (version: 99.99.99) is too new for your local Python client (version: {__version__}). "
        "Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }

    _, errors, _ = services.validate_model(
        params_as_dict=future_dict,
        validated_by=services.ValidationCalledBy.PIPELINE,
        root_item_type="Geometry",
    )

    assert errors[0] == {
<<<<<<< HEAD
        "type": "99.99.99 > 25.6.2b2",
=======
        "type": f"99.99.99 > {__version__}",
>>>>>>> 8d0cbbdd (Added version command and better project loading error (#1295))
        "loc": [],
        "msg": f"[Internal] Your `SimulationParams` (version: 99.99.99) is too new for the solver (version: {__version__}). Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Your `SimulationParams` (version: 99.99.99) is too new for the solver (version: {__version__}). Errors may occur since forward compatibility is limited."
        ),
    ):
        _, _, _ = services.generate_process_json(
            simulation_json=json.dumps(future_dict),
            up_to=CASE,
            root_item_type="Geometry",
        )


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
    services.change_unit_system(data=dict_to_convert, target_unit_system="Imperial")
    imperial_units = {"ft", "lbf", "lb", "s", "degF", "delta_degF", "rad", "degree"}

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
    services.change_unit_system(data=dict_to_convert, target_unit_system="CGS")
    CGS_units = {"dyn", "cm", "g", "s", "K", "rad", "degree"}

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
    services.change_unit_system(data=dict_to_convert, target_unit_system="SI")
    SI_units = {"m", "kg", "s", "K", "rad", "degree", "Pa"}

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
        == "[Internal] API misuse. Input version (999.999.999) is higher than requested target version (25.2.2)."
    )


def test_unit_conversion_front_end_compatibility():

    ##### 1. Ensure that the units are valid in `supported_units_by_front_end`
    def _get_all_units(value):
        if isinstance(value, dict):
            return [item for item in value.values()]
        else:
            assert isinstance(value, list)
            return value

    for dimension, value in supported_units_by_front_end.items():
        for unit in _get_all_units(value=value):
            if str(Unit(unit).dimensions) == dimension:
                continue
            elif (
                dimension == "(temperature_difference)"
                and str(Unit(unit).dimensions) == "(temperature)"
            ):
                continue
            else:
                raise ValueError(f"Unit {unit} is not valid for dimension {dimension}")

    ##### 2.  Ensure that all units supported have set their front-end approved units
    for field_name, field_info in _PredefinedUnitSystem.model_fields.items():
        if field_name == "name":
            continue
        print(">>> Unit: ", field_info.annotation.dim, field_info.annotation.dim.__class__)
        unit_system_dimension_string = str(field_info.annotation.dim)
        # for unit_name in unit:
        if unit_system_dimension_string not in supported_units_by_front_end.keys():
            raise ValueError(
                f"Unit {unit_system_dimension_string} (A.K.A {field_name}) is not supported by the front-end.",
                "Please ensure front end team is aware of this new unit and add its support.",
            )


def test_get_default_report_config_json():
    report_config_dict = services.get_default_report_config()
    with open("ref/default_report_config.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(report_config_dict, ref_dict, ignore_keys=["formatter"])
