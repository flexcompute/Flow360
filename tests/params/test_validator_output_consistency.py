import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.boundaries import (
    FreestreamBoundary,
    NoSlipWall,
    WallFunction,
)
from flow360.component.flow360_params.flow360_output import SurfaceOutput
from flow360.component.flow360_params.flow360_params import Flow360Params
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_consistency_wall_function_and_surface_output():
    with fl.SI_unit_system:
        param = Flow360Params(
            boundaries={"fluid/wing": NoSlipWall(), "fluid/farfield": FreestreamBoundary()},
            surface_output=SurfaceOutput(output_fields=["Cp"]),
        )

    with pytest.raises(
        ValueError,
        match="'wallFunctionMetric' in 'surfaceOutput' is only valid for 'WallFunction' boundary type",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                boundaries={"fluid/wing": NoSlipWall(), "fluid/farfield": FreestreamBoundary()},
                surface_output=SurfaceOutput(output_fields=["wallFunctionMetric"]),
            )


def test_consistency_ddes_and_volume_output():
    """
    todo: add check for DDES and volume output (beta feature)
    """
    return
