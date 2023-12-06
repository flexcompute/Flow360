import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamBoundary,
    HeatFluxWall,
    IsothermalWall,
    MassInflow,
    MassOutflow,
    MeshBoundary,
    NoSlipWall,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    TimeStepping,
    WallFunction,
)
from flow360.exceptions import ValidationError
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_mesh_boundary():
    mesh_boundary = MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            "fluid/fuselage",
            "fluid/leftWing",
            "fluid/rightWing"
        ]
    }
        """
    )
    assert mesh_boundary

    compare_to_ref(mesh_boundary, "../ref/flow360mesh/mesh_boundary/yaml.yaml")
    compare_to_ref(mesh_boundary, "../ref/flow360mesh/mesh_boundary/json.json")
    to_file_from_file_test(mesh_boundary)

    assert MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            1,
            2,
            3
        ]
    }
        """
    )


def test_case_boundary():
    with pytest.raises(ValueError):
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": TimeStepping(),
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            }
        )

    with pytest.raises(ValueError):
        param = Flow360Params.parse_raw(
            """
            {
                "boundaries": {
                    "fluid/fuselage": {
                        "type": "UnsupportedBC"
                    },
                    "fluid/leftWing": {
                        "type": "NoSlipWall"
                    },
                    "fluid/rightWing": {
                        "type": "NoSlipWall"
                    } 
                }
            }
            """
        )
        print(param)

    param = Flow360Params.parse_raw(
        """
        {
            "boundaries": {
                "fluid/fuselage": {
                    "type": "SlipWall"
                },
                "fluid/leftWing": {
                    "type": "NoSlipWall"
                },
                "fluid/rightWing": {
                    "type": "NoSlipWall"
                } 
            }
        }
        """
    )

    assert param

    boundaries = fl.Boundaries(
        wing=NoSlipWall(), symmetry=SlipWall(), freestream=FreestreamBoundary()
    )

    assert boundaries

    with pytest.raises(ValueError):
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": "NoSlipWall",
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            }
        )

    param = Flow360Params(
        boundaries={
            "fluid/fuselage": NoSlipWall(),
            "fluid/rightWing": NoSlipWall(),
            "fluid/leftWing": NoSlipWall(),
        }
    )

    assert param

    param = Flow360Params(
        boundaries={
            "fluid/ fuselage": NoSlipWall(),
            "fluid/rightWing": NoSlipWall(),
            "fluid/leftWing": SolidIsothermalWall(temperature=1.0),
        }
    )

    compare_to_ref(param.boundaries, "../ref/case_params/boundaries/yaml.yaml")
    compare_to_ref(param.boundaries, "../ref/case_params/boundaries/json.json")
    to_file_from_file_test(param)
    to_file_from_file_test(param.boundaries)

    SolidAdiabaticWall()
    SolidIsothermalWall(Temperature=10)

    with pytest.raises(pd.ValidationError):
        SolidIsothermalWall(Temperature=-1)


def test_boundary_incorrect():
    with pytest.raises(pd.ValidationError):
        MeshBoundary.parse_raw(
            """
            {
            "NoSlipWalls": [
                "fluid/fuselage",
                "fluid/leftWing",
                "fluid/rightWing"
            ]
        }
            """
        )


def test_boundary_types():
    assert NoSlipWall().type == "NoSlipWall"
    with fl.flow360_unit_system:
        assert NoSlipWall(velocity=(0, 0, 0))
    with fl.flow360_unit_system:
        assert NoSlipWall(name="name", velocity=[0, 0, 0])
    assert NoSlipWall(velocity=("0", "0.1*x+exp(y)+z^2", "cos(0.2*x*pi)+sqrt(z^2+1)"))
    assert SlipWall().type == "SlipWall"
    assert FreestreamBoundary().type == "Freestream"
    assert FreestreamBoundary(name="freestream")

    assert IsothermalWall(Temperature=1).type == "IsothermalWall"
    assert IsothermalWall(Temperature="exp(x)")

    assert HeatFluxWall(HeatFlux=-0.01).type == "HeatFluxWall"
    assert HeatFluxWall(HeatFlux="exp(x)", velocity=(0, 0, 0))

    assert SubsonicOutflowPressure(staticPressureRatio=1).type == "SubsonicOutflowPressure"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowPressure(staticPressureRatio=-1)

    assert SubsonicOutflowMach(Mach=1).type == "SubsonicOutflowMach"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowMach(Mach=-1)

    assert (
        SubsonicInflow(totalPressureRatio=1, totalTemperatureRatio=1, rampSteps=10).type
        == "SubsonicInflow"
    )
    assert SlidingInterfaceBoundary().type == "SlidingInterface"
    assert WallFunction().type == "WallFunction"
    assert MassInflow(massFlowRate=1).type == "MassInflow"
    with pytest.raises(pd.ValidationError):
        MassInflow(massFlowRate=-1)
    assert MassOutflow(massFlowRate=1).type == "MassOutflow"
    with pytest.raises(pd.ValidationError):
        MassOutflow(massFlowRate=-1)
