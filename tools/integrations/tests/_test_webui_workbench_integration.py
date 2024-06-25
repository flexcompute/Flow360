import json
import os

import flow360 as fl
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.services import (
    get_default_params,
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
)
from flow360.component.simulation.simulation_params import (
    MeshingParams,
    SimulationParams,
)
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import SI_unit_system, u

fl.UserConfig.set_profile("auto_test_1")
fl.Env.dev.active()

from flow360.component.geometry import Geometry
from flow360.examples import Airplane

SOLVER_VERSION = "workbench-24.6.0"


def get_all_process_jsons(params_as_dict):

    surface_json, hash = simulation_to_surface_meshing_json(
        params_as_dict, "SI", {"value": 1.0, "units": "m"}
    )
    print(surface_json)
    volume_json, hash = simulation_to_volume_meshing_json(
        params_as_dict, "SI", {"value": 1.0, "units": "m"}
    )
    print(volume_json)
    case_json, hash = simulation_to_case_json(params_as_dict, "SI", {"value": 1.0, "units": "m"})
    print(case_json)

    return surface_json, volume_json, case_json


with SI_unit_system:
    meshing = MeshingParams(
        refinements=[
            BoundaryLayer(first_layer_thickness=0.001),
            SurfaceRefinement(
                max_edge_length=0.15 * u.m,
                curvature_resolution_angle=10 * u.deg,
            ),
        ],
    )
    params = SimulationParams(
        meshing=meshing,
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
        ),
        operating_condition=AerospaceCondition(velocity_magnitude=100),
        models=[
            Wall(
                entities=[
                    Surface(name="fluid/rightWing"),
                    Surface(name="fluid/leftWing"),
                    Surface(name="fluid/fuselage"),
                ],
            ),
            Freestream(entities=[Surface(name="fluid/farfield")]),
        ],
        time_stepping=Steady(max_steps=700),
    )

with open("data/airplane_minimal_example_python.json", "w") as fh:
    params_as_dict = params.model_dump()
    json.dump(params_as_dict, fh, indent=4)


# run from params:
surface_json, volume_json, case_json = get_all_process_jsons(params.model_dump())
assert case_json["freestream"]["Mach"] == 0.2938635365101296


# run from full file:
with open("data/airplane_minimal_example_python.json") as fh:
    params_as_dict = json.load(fh)

surface_json, volume_json, case_json = get_all_process_jsons(params_as_dict)
assert case_json["freestream"]["Mach"] == 0.2938635365101296


# run from file without defaults:
with open("data/airplane_minimal_example_no_defaults.json") as fh:
    params_as_dict = json.load(fh)

surface_json, volume_json, case_json = get_all_process_jsons(params_as_dict)
assert case_json["freestream"]["Mach"] == 0.2938635365101296


# run from file without defaults:
with open("data/airplane_minimal_example_no_defaults_with_ids.json") as fh:
    params_as_dict = json.load(fh)

surface_json, volume_json, case_json = get_all_process_jsons(params_as_dict)
assert case_json["freestream"]["Mach"] == 0.2938635365101296
