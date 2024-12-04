from timeit import timeit

result = timeit(
    "from flow360.component.simulation.simulation_params import SimulationParams", number=10
)

print(result)
# tests on MacBook Pro M1:
# = 1.628892166 with wrap validator (this branch)
# = 1.437562916  without wrap validator (develop)


from flow360.component.simulation import services

data = services.get_default_params(
    unit_system_name="SI", length_unit="m", root_item_type="Geometry"
)


def run_validation(data):
    _, errors, _ = services.validate_model(params_as_dict=data, root_item_type="Geometry")


result = timeit(lambda: run_validation(data), number=10)

print(result)
# tests on MacBook Pro M1:
# = 0.051216541 with wrap validatorx
# = 0.615049542 without wrap validator
