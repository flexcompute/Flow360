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
    TimeStepping,
)

from flow360.exceptions import Flow360ConfigError, Flow360ValidationError
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_time_stepping():
    ts = TimeStepping()
    assert ts.json()
    to_file_from_file_test(ts)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping(physical_steps=10, time_step_size=-0.01)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping(physical_steps=10, time_step_size=(-0.01, "s"))

    ts = TimeStepping(time_step_size="inf")
    to_file_from_file_test(ts)

    ts = TimeStepping(physical_steps=10, time_step_size=0.001 * u.s)

    to_file_from_file_test(ts)

    assert ts.json()

    with fl.SI_unit_system:
        params = Flow360Params(
            geometry=Geometry(mesh_unit="mm", ref_area=1 * u.m**2),
            fluid_properties=fl.air,
            freestream=FreestreamFromVelocity(velocity=100 * u.m / u.s),
            time_stepping=ts,
        )

        assertions.assertAlmostEqual(
            json.loads(params.to_flow360_json())["timeStepping"]["timeStepSize"], 340.29400580821286
        )
        to_file_from_file_test(ts)

        params = Flow360Params(
            geometry={"meshUnit": "mm", "refArea": "m**2"},
            fluid_properties=fl.air,
            freestream={"temperature": 1, "Mach": 1, "mu_ref": 1},
            time_stepping=ts,
        )

    exported_json = json.loads(params.to_flow360_json())
    assert "meshUnit" not in exported_json["geometry"]

    ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3})
    assert ts.physical_steps == 3

    ts = TimeStepping.parse_obj({"physicalSteps": 2})
    assert ts.physical_steps == 2

    with pytest.raises(ValueError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physical_steps": 2})

    with pytest.raises(ValueError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physicalSteps": 2})


def test_time_stepping_cfl():
    cfl = fl.RampCFL(rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.AdaptiveCFL(min=0.1, max=2000, max_relative_change=1, convergence_limiting_factor=0.25)
    assert cfl
