import time
from copy import deepcopy

import pytest

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.services import (
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
)


def test_simulation_to_surface_meshing_json():
    param_data = {
        "meshing": {
            "defaults": {
                "curvature_resolution_angle": {"units": "degree", "value": 10.0},
                "surface_max_edge_length": {"units": "cm", "value": 15.0},
                "surface_edge_growth_rate": 1.07,
            },
            "refinements": [
                {
                    "entities": {
                        "stored_entities": [
                            {"name": "wingLeadingEdge"},
                            {"name": "wingTrailingEdge"},
                        ]
                    },
                    "method": {"type": "height", "value": {"units": "cm", "value": 0.03}},
                    "refinement_type": "SurfaceEdgeRefinement",
                },
                {
                    "entities": {
                        "stored_entities": [{"name": "rootAirfoilEdge"}, {"name": "tipAirfoilEdge"}]
                    },
                    "method": {"type": "projectAnisoSpacing"},
                    "refinement_type": "SurfaceEdgeRefinement",
                },
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x"],
                "face_group_tag": "not_used",
                "face_attribute_names": ["not_used"],
            }
        },
    }

    start_time = time.time()
    params, _, _ = validate_model(param_data, "SI", "Geometry", SURFACE_MESH)
    simulation_to_surface_meshing_json(params, {"value": 100.0, "units": "cm"})
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    with pytest.raises(ValueError, match="Mesh unit is required for translation."):
        simulation_to_surface_meshing_json(param_data, None)

    # TODO:  This needs more consideration. Is it allowed/possible to translate into an empty dict?
    # with pytest.raises(
    #     ValueError, match="No surface meshing parameters found in given SimulationParams."
    # ):
    #     simulation_to_surface_meshing_json(
    #         {
    #             "meshing": {"refinements": []},
    #             "unit_system": {"name": "SI"},
    #             "version": "24.2.0",
    #         },
    #         "SI",
    #         {"value": 100.0, "units": "cm"},
    #     )


def test_simulation_to_volume_meshing_json():
    param_data = {
        "meshing": {
            "refinement_factor": 1.45,
            "defaults": {
                "boundary_layer_first_layer_thickness": {"units": "m", "value": 1.35e-06},
                "boundary_layer_growth_rate": 1.04,
            },
            "refinements": [
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"units": "m", "value": [0.7, -1.0, 0.0]},
                                "height": {"units": "m", "value": 2.0},
                                "name": "cylinder_1",
                                "outer_radius": {"units": "m", "value": 1.1},
                            }
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "cm", "value": 7.5},
                },
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"units": "m", "value": [0.7, -1.0, 0.0]},
                                "height": {"units": "m", "value": 2.0},
                                "name": "cylinder_2",
                                "outer_radius": {"units": "m", "value": 2.2},
                            }
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "cm", "value": 10.0},
                },
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"units": "m", "value": [0.7, -1.0, 0.0]},
                                "height": {"units": "m", "value": 2.0},
                                "name": "cylinder_3",
                                "outer_radius": {"units": "m", "value": 3.3},
                            }
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "m", "value": 0.175},
                },
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"units": "m", "value": [0.7, -1.0, 0.0]},
                                "height": {"units": "m", "value": 2.0},
                                "name": "cylinder_4",
                                "outer_radius": {"units": "m", "value": 4.5},
                            }
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "mm", "value": 225.0},
                },
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "axis": [-1.0, 0.0, 0.0],
                                "center": {"units": "m", "value": [2.0, -1.0, 0.0]},
                                "height": {"units": "m", "value": 14.5},
                                "name": "outter_cylinder",
                                "outer_radius": {"units": "m", "value": 6.5},
                            }
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "mm", "value": 300.0},
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
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
    }

    params, _, _ = validate_model(param_data, "SI", "Geometry", VOLUME_MESH)

    sm_json, hash = simulation_to_volume_meshing_json(params, {"value": 100.0, "units": "cm"})
    assert sm_json["farfield"]["type"] == "auto"

    with pytest.raises(ValueError, match="Mesh unit is required for translation."):
        simulation_to_volume_meshing_json(params, None)


def test_simulation_to_case_json():
    param_data = {
        "models": [
            {
                "type": "Fluid",
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {"units": "K", "value": 111.0},
                        "reference_temperature": {"units": "K", "value": 273.0},
                        "reference_viscosity": {"units": "Pa*s", "value": 1.716e-05},
                    },
                    "name": "air",
                    "type": "air",
                },
                "navier_stokes_solver": {
                    "CFL_multiplier": 1.0,
                    "absolute_tolerance": 1e-10,
                    "equation_evaluation_frequency": 1,
                    "kappa_MUSCL": -1.0,
                    "limit_pressure_density": False,
                    "limit_velocity": False,
                    "linear_solver": {"max_iterations": 25},
                    "low_mach_preconditioner": False,
                    "max_force_jac_update_physical_steps": 0,
                    "type_name": "Compressible",
                    "numerical_dissipation_factor": 1.0,
                    "order_of_accuracy": 2,
                    "relative_tolerance": 0.0,
                    "update_jacobian_frequency": 4,
                },
                "turbulence_model_solver": {
                    "CFL_multiplier": 2.0,
                    "DDES": False,
                    "absolute_tolerance": 1e-08,
                    "equation_evaluation_frequency": 4,
                    "grid_size_for_LES": "maxEdgeLength",
                    "linear_solver": {"max_iterations": 15},
                    "max_force_jac_update_physical_steps": 0,
                    "modeling_constants": {
                        "C_DES": 0.72,
                        "C_d": 8.0,
                        "type_name": "SpalartAllmarasConsts",
                    },
                    "type_name": "SpalartAllmaras",
                    "order_of_accuracy": 2,
                    "quadratic_constitutive_relation": False,
                    "reconstruction_gradient_limiter": 0.5,
                    "relative_tolerance": 0.0,
                    "rotation_correction": False,
                    "update_jacobian_frequency": 4,
                },
            },
            {
                "entities": {"stored_entities": [{"name": "1"}]},
                "type": "Wall",
                "use_wall_function": False,
                "velocity": {"value": [0, 1, 2], "units": "m/s"},
            },
            {"entities": {"stored_entities": [{"name": "2"}]}, "type": "SlipWall"},
            {
                "entities": {"stored_entities": [{"name": "3"}]},
                "type": "Freestream",
            },
        ],
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "alpha": {"units": "degree", "value": 3.06},
            "beta": {"units": "degree", "value": 0.0},
            "thermal_state": {
                "density": {"units": "kg/m**3", "value": 1.225},
                "material": {
                    "dynamic_viscosity": {
                        "effective_temperature": {"units": "K", "value": 111.0},
                        "reference_temperature": {"units": "K", "value": 273.0},
                        "reference_viscosity": {"units": "Pa*s", "value": 1.716e-05},
                    },
                    "name": "air",
                    "type": "air",
                },
                "temperature": {"units": "K", "value": 288.15},
            },
            "velocity_magnitude": {"units": "m/s", "value": 288.12},
        },
        "outputs": [
            {
                "frequency": -1,
                "frequency_offset": 0,
                "output_fields": {
                    "items": ["primitiveVars", "residualNavierStokes", "residualTurbulence", "Mach"]
                },
                "output_format": "paraview",
                "output_type": "VolumeOutput",
            },
            {
                "entities": {
                    "stored_entities": [
                        {
                            "name": "sliceName_1",
                            "normal": [0.0, 1.0, 0.0],
                            "origin": {"units": "m", "value": [0.0, 0.56413, 0.0]},
                        }
                    ]
                },
                "frequency": -1,
                "frequency_offset": 0,
                "output_fields": {
                    "items": [
                        "primitiveVars",
                        "vorticity",
                        "T",
                        "s",
                        "Cp",
                        "mut",
                        "mutRatio",
                        "Mach",
                    ]
                },
                "output_format": "tecplot",
                "output_type": "SliceOutput",
            },
            {
                "entities": {"stored_entities": [{"name": "1"}, {"name": "2"}, {"name": "3"}]},
                "frequency": -1,
                "frequency_offset": 0,
                "output_fields": {"items": ["nuHat"]},
                "output_format": "paraview",
                "output_type": "SurfaceOutput",
                "write_single_file": False,
            },
            {
                "name": "my_integral",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "my_inflow1",
                        },
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "my_inflow2",
                        },
                    ]
                },
                "output_fields": {"items": ["primitiveVars", "vorticity"]},
                "output_type": "SurfaceIntegralOutput",
            },
            {
                "name": "my_probe",
                "entities": {
                    "stored_entities": [
                        {
                            "name": "DoesNotMatter1",
                            "private_attribute_entity_type_name": "Point",
                            "location": {"value": [1.0, 2.0, 3.0], "units": "ft"},
                        },
                        {
                            "name": "DoesNotMatter2",
                            "private_attribute_entity_type_name": "Point",
                            "location": {"value": [1.0, 2.0, 5.0], "units": "ft"},
                        },
                    ]
                },
                "output_fields": {"items": ["primitiveVars", "vorticity"]},
                "output_type": "ProbeOutput",
            },
        ],
        "reference_geometry": {
            "area": {"units": "m**2", "value": 0.748844455929999},
            "moment_center": {"units": "m", "value": [0.0, 0.0, 0.0]},
            "moment_length": {"units": "m", "value": 0.6460682372650963},
        },
        "time_stepping": {
            "CFL": {"final": 200.0, "initial": 5.0, "ramp_steps": 40, "type": "ramp"},
            "max_steps": 2000,
            "type_name": "Steady",
            "order_of_accuracy": 2,
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
    }

    params, _, _ = validate_model(param_data, "SI", "Geometry")

    simulation_to_case_json(params, {"value": 100.0, "units": "cm"})

    with pytest.raises(ValueError, match="Mesh unit is required for translation."):
        simulation_to_case_json(param_data, None)


def test_simulation_to_all_translation():
    with SI_unit_system:
        meshing = MeshingParams(
            defaults=MeshingDefaults(
                surface_edge_growth_rate=1.5,
                boundary_layer_first_layer_thickness=0.001,
                curvature_resolution_angle=10 * u.deg,
                surface_max_edge_length=15 * u.cm,
            ),
            refinements=[],
            volume_zones=[AutomatedFarfield()],
        )
        param = SimulationParams(
            meshing=meshing,
            reference_geometry=ReferenceGeometry(
                moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
            ),
            operating_condition=AerospaceCondition(velocity_magnitude=100),
            models=[
                Fluid(),
                Wall(
                    name="wall0",
                    entities=[Surface(name="wing1"), Surface(name="wing2")],
                ),
                Freestream(entities=[Surface(name="farfield")]),
            ],
            private_attribute_asset_cache=AssetCache(
                project_entity_info=GeometryEntityInfo(
                    face_group_tag="not_used",
                    face_ids=["face_x"],
                    face_attribute_names=["not_used"],
                )
            ),
        )

    surface_json, hash = simulation_to_surface_meshing_json(param, {"value": 100.0, "units": "cm"})
    print(surface_json)
    volume_json, hash = simulation_to_volume_meshing_json(param, {"value": 100.0, "units": "cm"})
    print(volume_json)
    case_json, hash = simulation_to_case_json(param, {"value": 100.0, "units": "cm"})
    print(case_json)


def test_simulation_to_case_vm_workflow():
    param_data = {
        "meshing": {"wrong": "parameters"},
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "velocity_magnitude": {"value": 100, "units": "m/s"},
            "alpha": {"value": 0, "units": "degree"},
            "beta": {"value": 0, "units": "degree"},
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
    }

    params, _, _ = validate_model(param_data, "SI", "Geometry")

    with pytest.raises(ValueError):
        case_json, hash = simulation_to_case_json(params, {"value": 100.0, "units": "cm"})
        print(case_json)

    params, _, _ = validate_model(param_data, "SI", "VolumeMesh")
    case_json, hash = simulation_to_case_json(params, {"value": 100.0, "units": "cm"})
    print(case_json)


def test_simulation_to_all_translation_2():
    params_as_dict = {
        "meshing": {
            "refinement_factor": 1,
            "defaults": {
                "surface_edge_growth_rate": 1.2,
                "boundary_layer_first_layer_thickness": {"value": 0.001, "units": "m"},
                "boundary_layer_growth_rate": 1.2,
                "curvature_resolution_angle": {"value": 10, "units": "degree"},
                "surface_max_edge_length": {"value": 0.15, "units": "m"},
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
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "velocity_magnitude": {"value": 100, "units": "m/s"},
            "alpha": {"value": 0, "units": "degree"},
            "beta": {"value": 0, "units": "degree"},
        },
        "reference_geometry": {
            "moment_center": {"value": [0, 0, 0], "units": "m"},
            "moment_length": {"value": 1, "units": "m"},
            "area": {"value": 1, "units": "m**2"},
        },
        "models": [],
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x"],
                "face_group_tag": "not_used",
                "face_attribute_names": ["not_used"],
            }
        },
    }

    params, _, _ = validate_model(params_as_dict, "SI", "Geometry")

    surface_json, hash = simulation_to_surface_meshing_json(params, {"value": 100.0, "units": "cm"})
    print(surface_json)
    volume_json, hash = simulation_to_volume_meshing_json(params, {"value": 100.0, "units": "cm"})
    print(volume_json)
    case_json, hash = simulation_to_case_json(params, {"value": 100.0, "units": "cm"})
    print(case_json)
