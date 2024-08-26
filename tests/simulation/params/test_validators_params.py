import re
import unittest

import pytest

from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system

assertions = unittest.TestCase("__init__")


@pytest.fixture()
def surface_output_with_wall_metric():
    surface_output = SurfaceOutput(
        name="surface", write_single_file=True, output_fields=["wallFunctionMetric"]
    )
    return surface_output


@pytest.fixture()
def wall_model_with_function():
    wall_model = Wall(name="wall", surfaces=[Surface(name="noSlipWall")], use_wall_function=True)
    return wall_model


@pytest.fixture()
def wall_model_without_function():
    wall_model = Wall(name="wall", surfaces=[Surface(name="noSlipWall")], use_wall_function=False)
    return wall_model


def test_consistency_wall_function_validator(
    surface_output_with_wall_metric, wall_model_with_function, wall_model_without_function
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[wall_model_with_function], outputs=[surface_output_with_wall_metric]
        )

    assert params

    message = "To use 'wallFunctionMetric' for output specify a Wall with use_wall_function=true"

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[wall_model_without_function], outputs=[surface_output_with_wall_metric]
        )
