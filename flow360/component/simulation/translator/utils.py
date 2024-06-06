from __future__ import annotations
from flow360.component.simulation.simulation_params import SimulationParams  # Not required
import json
import functools


def preprocess_input(func):
    @functools.wraps(func)
    def wrapper(input_params, *args, **kwargs):
        processed_input = get_simulation_param_dict(input_params)
        return func(processed_input, *args, **kwargs)

    return wrapper


def get_simulation_param_dict(input_params: SimulationParams | str | dict):
    """
    Get the dictionary of `SimulationParams`.
    """
    if isinstance(input_params, SimulationParams):
        # pylint: disable=fixme
        # TODO:  1. unit test that processed param.preprocess return itself.
        # TODO:  2. preprocess() does not have units conversion so the seralized obj still has dimension. Therefore we cannot
        # TODO:  test this path for now.
        processed_param = input_params.preprocess()
        return processed_param.model_dump()
    if isinstance(input_params, str):
        try:
            # If input is a json string
            return json.loads(input_params)
        except json.JSONDecodeError as e:
            # If input is a file path
            with open(input_params, "r") as file:
                return json.load(file)

    if isinstance(input_params, dict):
        return input_params

    raise ValueError(f"Invalid input <{input_params.__class__.__name__}> for translator. ")
