import unittest

import pytest

import flow360 as fl
from flow360.component.flow360_params.boundaries import (
    NoSlipWall,
    TranslationallyPeriodic,
)
from flow360.component.flow360_params.flow360_output import AeroacousticOutput
from flow360.component.flow360_params.flow360_params import Flow360Params
from globals.flags import Flags

if Flags.beta_features():
    pass

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_aero_acoustics():
    """
    todo: warning for inaccurate simulations
    """
    with fl.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/right"),
                "blk-1/right": TranslationallyPeriodic(),
            }
        )
        param = Flow360Params(
            aeroacoustic_output=AeroacousticOutput(observers=[(1, 2, 3), (4, 5, 6)]),
            boundaries={
                "blk-1/right": NoSlipWall(),
            },
        )
