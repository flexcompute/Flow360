import unittest

import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    SteadyTimeStepping,
)
from flow360.component.flow360_params.solvers import SpalartAllmaras
from flow360.component.flow360_params.time_stepping import UnsteadyTimeStepping
from globals.flags import Flags

if Flags.beta_features():
    pass

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_consistency_ddes_unsteady():
    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=UnsteadyTimeStepping(),
            turbulence_model_solver=SpalartAllmaras(DDES=True),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=SteadyTimeStepping(),
            turbulence_model_solver=SpalartAllmaras(),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
    with fl.SI_unit_system:
        param = Flow360Params(
            turbulence_model_solver=SpalartAllmaras(),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with pytest.raises(
        ValueError,
        match="Running DDES with steady simulation is invalid.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=SteadyTimeStepping(),
                turbulence_model_solver=SpalartAllmaras(DDES=True),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

    with pytest.raises(
        ValueError,
        match="Running DDES with steady simulation is invalid.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                turbulence_model_solver=SpalartAllmaras(DDES=True),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
