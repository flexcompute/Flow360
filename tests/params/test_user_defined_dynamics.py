import unittest

import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import UserDefinedDynamic
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_user_defined_dynamics():
    udd = UserDefinedDynamic(
        name="alphaController",
        input_vars=["CL", "rotMomentX"],
        constants={"CLTarget": 0.4, "Kp": 0.2, "Ki": 0.002},
        output_vars={"alphaAngle": "if (pseudoStep > 500) state[0]; else alphaAngle;"},
        state_vars_initial_value=["alphaAngle", "rotMomentY"],
        update_law=[
            "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
            "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
        ],
        output_target_name="target",
        inputBoundaryPatches=["fluid/wall"],
    )

    assert udd

    to_file_from_file_test(udd)

    with fl.SI_unit_system:
        param = fl.Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
            time_stepping=fl.SteadyTimeStepping(max_pseudo_steps=441),
            user_defined_dynamics=[udd],
        )
        solver_params = param.to_solver()
        assert solver_params.user_defined_dynamics[0].input_vars == ["CL", "momentX"]
        assert (
            solver_params.user_defined_dynamics[0].output_vars["alphaAngle"]
            == "(pseudoStep > 500) ? (state[0]) : (alphaAngle);"
        )
        assert solver_params.user_defined_dynamics[0].state_vars_initial_value == [
            "alphaAngle",
            "momentY",
        ]
        assert solver_params.user_defined_dynamics[0].update_law == [
            "(pseudoStep > 500) ? (state[0] + Kp * (CLTarget - CL) + Ki * state[1]) : (state[0]);",
            "(pseudoStep > 500) ? (state[1] + (CLTarget - CL)) : (state[1]);",
        ]
