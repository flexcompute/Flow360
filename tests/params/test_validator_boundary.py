import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.boundaries import (
    FreestreamBoundary,
    HeatFluxWall,
    IsothermalWall,
    MassInflow,
    MassOutflow,
    NoSlipWall,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    SupersonicInflow,
    WallFunction,
)
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    MeshBoundary,
    SteadyTimeStepping,
)
from flow360.component.flow360_params.turbulence_quantities import TurbulenceQuantities
from flow360.exceptions import Flow360ValidationError
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_duplidated_boundary_names():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match="Boundary name <wing> under patch <fluid/.*Wing> appears multiple times",
        ):
            param = Flow360Params(
                boundaries={
                    "fluid/fuselage": NoSlipWall(name="fuselage"),
                    "fluid/leftWing": NoSlipWall(name="wing"),
                    "fluid/rightWing": NoSlipWall(name="wing"),
                }
            )


def test_tri_quad_boundaries():
    """
    todo: handle warning
    """
    with fl.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "fluid/tri_fuselage": NoSlipWall(),
                "fluid/quad_fuselage": NoSlipWall(),
                "fluid/tri_wing": NoSlipWall(),
                "fluid/quad_wing": NoSlipWall(),
            }
        )
