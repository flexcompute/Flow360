import json
import math
import re
import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    AeroacousticOutput,
    Flow360MeshParams,
    Flow360Params,
    FluidDynamicsVolumeZone,
    ForcePerArea,
    Freestream,
    FreestreamBoundary,
    Geometry,
    HeatEquationSolver,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    IsothermalWall,
    LinearSolver,
    MassInflow,
    MassOutflow,
    MeshBoundary,
    MeshSlidingInterface,
    NavierStokesSolver,
    NoSlipWall,
    ReferenceFrame,
    SlidingInterface,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    TimeStepping,
    VolumeZones,
    WallFunction,
)
from flow360.component.types import TimeStep
from flow360.exceptions import ConfigError, ValidationError

from .utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_flow360meshparam():
    mp0 = Flow360MeshParams.parse_raw(
        """
    {
        "boundaries": {
            "noSlipWalls": [
                "fluid/fuselage",
                "fluid/leftWing",
                "fluid/rightWing"
            ]
        }
    }
    """
    )
    assert mp0
    to_file_from_file_test(mp0)

    mp1 = Flow360MeshParams.parse_raw(
        """
        {
        "boundaries": {
            "noSlipWalls": [
                1,
                2,
                3
            ]
        }
    }
        """
    )

    assert mp1
    to_file_from_file_test(mp1)

    mp2 = Flow360MeshParams(
        boundaries=MeshBoundary(
            no_slip_walls=["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]
        )
    )
    assert mp2
    assert mp0 == mp2
    to_file_from_file_test(mp2)


def test_flow360param():
    mesh = Flow360Params.parse_raw(
        """
        {
    "boundaries": {
        "fluid/fuselage": {
            "type": "NoSlipWall"
        },
        "fluid/leftWing": {
            "type": "NoSlipWall"
        },
        "fluid/rightWing": {
            "type": "NoSlipWall"
        },
        "fluid/farfield": {
            "type": "Freestream"
        }
    },
    "actuatorDisks": [
        {
            "center": [
                3.6,
                -5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                -0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        },
        {
            "center": [
                3.6,
                5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        }
    ],
    "freestream": {"temperature": 1, "Mach": 0.5}
}
        """
    )

    assert mesh


def test_flow360param1():
    params = Flow360Params(freestream=Freestream.from_speed(10))
    assert params.time_stepping.max_pseudo_steps is None
    params.time_stepping = TimeStepping(physical_steps=100)
    assert params


def test_tuple_from_yaml():
    fs = Freestream("data/case_params/freestream/yaml.yaml")
    assert fs


def test_update_from_multiple_files():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    params.append(outputs)

    assert params
    to_file_from_file_test(params)
    compare_to_ref(params, "ref/case_params/params.yaml")
    compare_to_ref(params, "ref/case_params/params.json", content_only=True)


def test_update_from_multiple_files_dont_overwrite():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    outputs.geometry = fl.Geometry(ref_area=2)
    params.append(outputs)

    assert params.geometry.ref_area == 1.15315084119231


def test_update_from_multiple_files_overwrite():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    outputs.geometry = fl.Geometry(ref_area=2)
    params.append(outputs, overwrite=True)

    assert params.geometry.ref_area == 2


def clear_formatting(message):
    # Remove color formatting escape codes
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    cleared = ansi_escape.sub("", message).replace("\n", "")
    cleared = re.sub(r" +", " ", cleared)
    return cleared


def test_depracated(capfd):
    ns = fl.NavierStokesSolver(tolerance=1e-8)
    captured = capfd.readouterr()
    expected = f'WARNING: "tolerance" is deprecated. Use "absolute_tolerance" OR "absoluteTolerance" instead'
    assert expected in clear_formatting(captured.out)

    ns = fl.TimeStepping(maxPhysicalSteps=10)
    captured = capfd.readouterr()
    expected = f'WARNING: "maxPhysicalSteps" is deprecated. Use "physical_steps" OR "physicalSteps" instead'
    assert expected in clear_formatting(captured.out)
