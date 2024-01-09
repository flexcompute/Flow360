import unittest

import pytest

from flow360.component.flow360_params.flow360_params import UserDefinedDynamic
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_user_defined_dynamics():
    udd = UserDefinedDynamic(
        name="alphaController",
        input_vars=["CL"],
        constants={"CLTarget": 0.4, "Kp": 0.2, "Ki": 0.002},
        output_vars={"alphaAngle": "if (pseudoStep > 500) state[0]; else alphaAngle;"},
        state_vars_initial_value=["alphaAngle", "0.0"],
        update_law=[
            "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
            "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
        ],
        output_target_name="target",
        output_law=[
            "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
            "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
        ],
        inputBoundaryPatches=["fluid/wall"],
    )

    assert udd

    to_file_from_file_test(udd)
