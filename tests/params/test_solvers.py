import os
import unittest

import pydantic as pd
import pytest

import flow360 as fl
from config.flags import Flags
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    HeatEquationSolver,
    KOmegaSST,
    LinearSolver,
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolver,
)
from flow360.component.flow360_params.solvers import NavierStokesSolver
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_navier_stokes():
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappa_MUSCL=-2)
    assert NavierStokesSolver(kappa_MUSCL=-1)
    assert NavierStokesSolver(kappa_MUSCL=1)
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappa_MUSCL=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=0)

    assert NavierStokesSolver(order_of_accuracy=1)
    assert NavierStokesSolver(order_of_accuracy=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=3)

    ns = NavierStokesSolver(
        absolute_tolerance=1e-10,
        kappa_MUSCL=-1,
        relative_tolerance=0,
        CFL_multiplier=1,
        linear_iterations=30,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        limit_velocity=True,
        limit_pressure_density=False,
    )
    with fl.SI_unit_system:
        p = Flow360Params(
            navier_stokes_solver=ns,
            boundaries={},
            freestream={"modelType": "FromMach", "Mach": 1, "Temperature": 1, "muRef": 1},
        )
    to_file_from_file_test(p)


def test_turbulence_solver():
    ts = SpalartAllmaras()
    assert ts
    ts = KOmegaSST()
    assert ts
    ts = NoneSolver()
    assert ts

    ts = SpalartAllmaras(
        absolute_tolerance=1e-10,
        relative_tolerance=0,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        DDES=True,
        grid_size_for_LES="maxEdgeLength",
        model_constants={"C_DES": 0.85, "C_d": 8.0},
    )
    to_file_from_file_test(ts)


def test_transition():
    tr = TransitionModelSolver()
    assert tr

    tr = TransitionModelSolver(
        update_jacobian_frequency=5,
        equation_eval_frequency=10,
        max_force_jac_update_physical_steps=10,
        order_of_accuracy=1,
        turbulence_intensity_percent=1.2,
        N_crit=2,
    )
    to_file_from_file_test(tr)


def test_heat_equation():
    he = HeatEquationSolver(
        equation_eval_frequency=10,
        linear_solver_config=LinearSolver(
            absoluteTolerance=1e-10,
            max_iterations=50,
        ),
    )
    assert he

    if Flags.beta_features():
        compare_to_ref(he, "../ref/case_params/heat_equation/ref.json", content_only=True)
