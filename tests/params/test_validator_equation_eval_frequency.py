import os
import unittest

import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    SteadyTimeStepping,
)
from flow360.component.flow360_params.solvers import (
    SpalartAllmaras,
    TransitionModelSolver,
)
from flow360.component.flow360_params.time_stepping import UnsteadyTimeStepping

if os.environ.get("FLOW360_BETA_FEATURES", False):
    pass

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_equation_eval_frequency_for_unsteady_simulations():
    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=UnsteadyTimeStepping(max_pseudo_steps=30),
            turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=2),
            transition_model_solver=TransitionModelSolver(equation_eval_frequency=4),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=SteadyTimeStepping(max_pseudo_steps=10),
            turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=12),
            transition_model_solver=TransitionModelSolver(equation_eval_frequency=15),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with pytest.raises(
        ValueError,
        match="'equation evaluation frequency' in turbulence_model_solver is greater than max_pseudo_steps.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(max_pseudo_steps=2),
                turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=3),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
    with pytest.raises(
        ValueError,
        match="'equation evaluation frequency' in transition_model_solver is greater than max_pseudo_steps.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(max_pseudo_steps=2),
                transition_model_solver=TransitionModelSolver(equation_eval_frequency=3),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
