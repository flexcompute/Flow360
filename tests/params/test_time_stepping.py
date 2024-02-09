import json
import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360 import units as u
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    SteadyTimeStepping,
    UnsteadyTimeStepping,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_time_stepping():
    ts = SteadyTimeStepping()
    assert ts.json()
    to_file_from_file_test(ts)

    with pytest.raises(pd.ValidationError):
        ts = UnsteadyTimeStepping(physical_steps=10, time_step_size=-0.01)

    with pytest.raises(pd.ValidationError):
        ts = UnsteadyTimeStepping(physical_steps=10, time_step_size=(-0.01, "s"))

    ts = SteadyTimeStepping(time_step_size="inf")
    to_file_from_file_test(ts)

    ts = UnsteadyTimeStepping(physical_steps=10, time_step_size=0.001 * u.s)

    to_file_from_file_test(ts)

    assert ts.json()

    with fl.SI_unit_system:
        params = Flow360Params(
            geometry=Geometry(mesh_unit="mm", ref_area=1 * u.m**2),
            fluid_properties=fl.air,
            boundaries={},
            freestream=FreestreamFromVelocity(velocity=100 * u.m / u.s),
            time_stepping=ts,
        )

        assertions.assertAlmostEqual(
            json.loads(params.flow360_json())["timeStepping"]["timeStepSize"], 340.29400580821286
        )
        to_file_from_file_test(ts)

        params = Flow360Params(
            geometry={"meshUnit": "mm", "refArea": "m**2"},
            boundaries={},
            fluid_properties=fl.air,
            freestream={"modelType": "FromMach", "temperature": 288.15, "Mach": 1, "mu_ref": 1},
            time_stepping=ts,
        )

    exported_json = json.loads(params.flow360_json())
    assert "meshUnit" not in exported_json["geometry"]

    ts = UnsteadyTimeStepping.parse_obj({"maxPhysicalSteps": 3})
    assert ts.physical_steps == 3

    ts = UnsteadyTimeStepping.parse_obj({"physicalSteps": 2})
    assert ts.physical_steps == 2

    with pytest.raises(ValueError):
        ts = UnsteadyTimeStepping.parse_obj({"maxPhysicalSteps": 3, "physical_steps": 2})

    with pytest.raises(ValueError):
        ts = UnsteadyTimeStepping.parse_obj({"maxPhysicalSteps": 3, "physicalSteps": 2})

    ## Tests for default values
    def assert_steady_Ramp(ts):
        assert ts.CFL.initial == fl.RampCFL().initial
        assert ts.CFL.final == fl.RampCFL().final
        assert ts.CFL.ramp_steps == fl.RampCFL().ramp_steps

    def assert_unsteady_Ramp(ts):
        assert ts.CFL.initial == fl.RampCFL.default_unsteady().initial
        assert ts.CFL.final == fl.RampCFL.default_unsteady().final
        assert ts.CFL.ramp_steps == fl.RampCFL.default_unsteady().ramp_steps

    def assert_steady_Adaptive(ts):
        assert ts.CFL.min == fl.AdaptiveCFL().min
        assert ts.CFL.max == fl.AdaptiveCFL().max
        assert ts.CFL.max_relative_change == fl.AdaptiveCFL().max_relative_change
        assert ts.CFL.convergence_limiting_factor == fl.AdaptiveCFL().convergence_limiting_factor

    def assert_unsteady_Adaptive(ts):
        assert ts.CFL.min == fl.AdaptiveCFL.default_unsteady().min
        assert ts.CFL.max == fl.AdaptiveCFL.default_unsteady().max
        assert ts.CFL.max_relative_change == fl.AdaptiveCFL.default_unsteady().max_relative_change
        assert (
            ts.CFL.convergence_limiting_factor
            == fl.AdaptiveCFL.default_unsteady().convergence_limiting_factor
        )

    ts = SteadyTimeStepping()
    assert_steady_Adaptive(ts)
    ts = SteadyTimeStepping(CFL=fl.RampCFL())
    assert_steady_Ramp(ts)
    ts = SteadyTimeStepping(CFL=fl.AdaptiveCFL())
    assert_steady_Adaptive(ts)

    ts = UnsteadyTimeStepping()
    assert_unsteady_Adaptive(ts)
    ts = UnsteadyTimeStepping(CFL=fl.RampCFL())
    assert_unsteady_Ramp(ts)
    ts = UnsteadyTimeStepping(CFL=fl.AdaptiveCFL())
    assert_unsteady_Adaptive(ts)


def test_time_stepping_cfl():
    cfl = fl.RampCFL(rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.AdaptiveCFL(min=0.1, max=2000, max_relative_change=1, convergence_limiting_factor=0.25)
    assert cfl
