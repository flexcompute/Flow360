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
    def assert_CFL(ts, trueValue):
        for field_name in ts.CFL.__fields__:
            assert getattr(ts.CFL, field_name) == getattr(trueValue, field_name)

    ts = SteadyTimeStepping()
    assert_CFL(ts, fl.AdaptiveCFL.default_steady())
    ts = SteadyTimeStepping(CFL=fl.RampCFL())
    assert_CFL(ts, fl.RampCFL.default_steady())
    ts = SteadyTimeStepping(CFL=fl.RampCFL(final=1000))
    assert_CFL(ts, fl.RampCFL(initial=5, final=1000, ramp_steps=40))
    ts = SteadyTimeStepping(CFL=fl.AdaptiveCFL())
    assert_CFL(ts, fl.AdaptiveCFL.default_steady())
    ts = SteadyTimeStepping(CFL=fl.AdaptiveCFL(maxRelativeChange=1000))
    assert_CFL(
        ts, fl.AdaptiveCFL(max=1e4, convergence_limiting_factor=0.25, max_relative_change=1000)
    )

    ts = UnsteadyTimeStepping()
    assert_CFL(ts, fl.AdaptiveCFL.default_unsteady())
    ts = UnsteadyTimeStepping(CFL=fl.RampCFL())
    assert_CFL(ts, fl.RampCFL.default_unsteady())
    ts = UnsteadyTimeStepping(CFL=fl.RampCFL(final=1000))
    assert_CFL(ts, fl.RampCFL(initial=1, final=1000, ramp_steps=30))
    ts = UnsteadyTimeStepping(CFL=fl.AdaptiveCFL())
    assert_CFL(ts, fl.AdaptiveCFL.default_unsteady())
    ts = UnsteadyTimeStepping(CFL=fl.AdaptiveCFL(maxRelativeChange=1000))
    assert_CFL(
        ts, fl.AdaptiveCFL(max=1e6, convergence_limiting_factor=1.0, max_relative_change=1000)
    )
    ts = UnsteadyTimeStepping(CFL=fl.AdaptiveCFL(maxRelativeChange=1000))
    assert_CFL(
        ts,
        fl.AdaptiveCFL(max=1e6, convergence_limiting_factor=1.0, max_relative_change=1000),
    )


def test_time_stepping_cfl():
    cfl = fl.RampCFL(rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.AdaptiveCFL(min=0.1, max=2000, max_relative_change=1, convergence_limiting_factor=0.25)
    assert cfl
