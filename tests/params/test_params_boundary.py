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
from flow360.exceptions import Flow360ValidationError
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
    with fl.SI_unit_system:
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
                        "modelType": "SlipWall"
                    },
                    "fluid/leftWing": {
                        "modelType": "NoSlipWall"
                    },
                    "fluid/rightWing": {
                        "modelType": "NoSlipWall"
                    } 
                }
            }
            """
        )

        param = Flow360Params.parse_raw(
            """
            {
                "boundaries": {
                    "fluid/fuselage": {
                        "modelType": "SlipWall"
                    },
                    "fluid/leftWing": {
                        "modelType": "NoSlipWall"
                    },
                    "fluid/rightWing": {
                        "modelType": "NoSlipWall"
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
    assert NoSlipWall().model_type == "NoSlipWall"
    with fl.flow360_unit_system:
        assert NoSlipWall(velocity=(0, 0, 0))
    with fl.flow360_unit_system:
        assert NoSlipWall(name="name", velocity=[0, 0, 0])
    assert NoSlipWall(velocity=("0", "0.1*x+exp(y)+z^2", "cos(0.2*x*pi)+sqrt(z^2+1)"))
    assert SlipWall().model_type == "SlipWall"
    assert FreestreamBoundary().model_type == "Freestream"
    assert FreestreamBoundary(name="freestream")

    assert IsothermalWall(Temperature=1).model_type == "IsothermalWall"
    assert IsothermalWall(Temperature="exp(x)")

    assert HeatFluxWall(heatFlux=-0.01).model_type == "HeatFluxWall"
    with fl.flow360_unit_system:
        assert HeatFluxWall(heatFlux="exp(x)", velocity=(0, 0, 0))

    assert SubsonicOutflowPressure(staticPressureRatio=1).model_type == "SubsonicOutflowPressure"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowPressure(staticPressureRatio=-1)

    assert SubsonicOutflowMach(Mach=1).model_type == "SubsonicOutflowMach"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowMach(Mach=-1)

    assert (
        SubsonicInflow(totalPressureRatio=1, totalTemperatureRatio=1, rampSteps=10).model_type
        == "SubsonicInflow"
    )
    assert SlidingInterfaceBoundary().model_type == "SlidingInterface"
    assert WallFunction().model_type == "WallFunction"
    assert MassInflow(massFlowRate=1).model_type == "MassInflow"
    with pytest.raises(pd.ValidationError):
        MassInflow(massFlowRate=-1)
    assert MassOutflow(massFlowRate=1).model_type == "MassOutflow"
    with pytest.raises(pd.ValidationError):
        MassOutflow(massFlowRate=-1)
