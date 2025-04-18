import cProfile
import json
import time

from flow360.component.simulation.services import (
    ValidationCalledBy,
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
    validate_model,
)

start_time_glb = time.time()
start_time = time.time()
with open("./data/large_simulation.json", "r") as f:
    params_as_dict = json.load(f)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds [json.load()]")
start_time = time.time()
params, _, _ = validate_model(
    params_as_dict=params_as_dict, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds [validate_model]")
_, hash = simulation_to_surface_meshing_json(params, {"value": 100.0, "units": "cm"})
_, hash = simulation_to_volume_meshing_json(params, {"value": 100.0, "units": "cm"})
_, hash = simulation_to_case_json(params, {"value": 100.0, "units": "cm"})
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds [translation Case+SM+VM]")


def translation_wrapper():
    simulation_to_surface_meshing_json(params, {"value": 100.0, "units": "cm"})


def validation_wrapper():
    _, _, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )


cProfile.run("translation_wrapper()", "profile_translator.prof")
cProfile.run("validation_wrapper()", "profile_validation.prof")
