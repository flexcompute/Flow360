import json
import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    Freestream,
    Geometry,
    TimeStepping,
)
from flow360.component.types import TimeStep
from flow360.exceptions import ConfigError, ValidationError

from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_time_stepping():
    ts = TimeStepping.default_steady()
    assert ts.json()
    assert ts.to_flow360_json()
    to_file_from_file_test(ts)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping.default_unsteady(physical_steps=10, time_step_size=-0.01)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping.default_unsteady(physical_steps=10, time_step_size=(-0.01, "s"))

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping.default_unsteady(physical_steps=10, time_step_size="infinity")

    ts = TimeStepping(time_step_size="inf")
    to_file_from_file_test(ts)

    ts = TimeStepping.default_unsteady(physical_steps=10, time_step_size=(0.01, "s"))
    assert isinstance(ts.time_step_size, TimeStep)

    to_file_from_file_test(ts)

    assert ts.json()
    with pytest.raises(ConfigError):
        ts.to_flow360_json()

    assert ts.to_flow360_json(mesh_unit_length=0.2, C_inf=2)

    params = Flow360Params(
        geometry=Geometry(mesh_unit="mm"), freestream=Freestream.from_speed(10), time_stepping=ts
    )

    assertions.assertAlmostEqual(
        json.loads(params.to_flow360_json())["timeStepping"]["timeStepSize"], 0.1
    )
    to_file_from_file_test(ts)

    params = Flow360Params(
        geometry={"meshUnit": "mm"},
        freestream={"temperature": 1, "Mach": 1, "density": 1},
        time_stepping=ts,
    )
    exported_json = json.loads(params.to_flow360_json())
    assert "meshUnit" not in exported_json["geometry"]

    ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3})
    assert ts.physical_steps == 3

    ts = TimeStepping.parse_obj({"physicalSteps": 2})
    assert ts.physical_steps == 2

    with pytest.raises(ValidationError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physical_steps": 2})

    with pytest.raises(ValidationError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physicalSteps": 2})


def test_time_stepping_cfl():
    cfl = fl.TimeSteppingCFL(rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.TimeSteppingCFL(type="ramp", rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.TimeSteppingCFL(
        type="adaptive", min=0.1, max=2000, max_relative_change=1, convergence_limiting_factor=0.25
    )
    assert cfl

    cfl = fl.TimeSteppingCFL.adaptive()
    assert cfl
