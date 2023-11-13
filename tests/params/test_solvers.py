import unittest

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    HeatEquationSolver,
    LinearSolver,
    NavierStokesSolver,
    TransitionModelSolver,
    TurbulenceModelSolverNone,
    TurbulenceModelSolverSA,
    TurbulenceModelSolverSST,
)
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_navier_stokes():
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappaMUSCL=-2)
    assert NavierStokesSolver(kappaMUSCL=-1)
    assert NavierStokesSolver(kappaMUSCL=1)
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappaMUSCL=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=0)

    assert NavierStokesSolver(order_of_accuracy=1)
    assert NavierStokesSolver(order_of_accuracy=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=3)

    ns = NavierStokesSolver(
        absolute_tolerance=1e-10,
        kappaMUSCL=-1,
        relative_tolerance=0,
        CFL_multiplier=1,
        linear_iterations=30,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        limit_velocity=True,
        limit_pressure_density=False,
        numerical_dissipation_factor=0.2,
    )
    p = Flow360Params(
        navier_stokes_solver=ns,
        freestream={"Mach": 1, "Temperature": 1},
    )
    to_file_from_file_test(p)


def test_turbulence_solver():
    ts = TurbulenceModelSolverSA()
    assert ts
    ts = TurbulenceModelSolverSST()
    assert ts
    ts = TurbulenceModelSolverNone()
    assert ts

    ts = TurbulenceModelSolverSA(
        absolute_tolerance=1e-10,
        relative_tolerance=0,
        linear_iterations=30,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        DDES=True,
        grid_size_for_LES="maxEdgeLength",
        model_constants={"C_DES1": 0.85, "C_d1": 8.0},
    )
    to_file_from_file_test(ts)


def test_transition():
    tr = TransitionModelSolver()
    assert tr

    tr = TransitionModelSolver(
        CFL_multiplier=0.5,
        linear_iterations=10,
        update_jacobian_frequency=5,
        equation_eval_frequency=10,
        max_force_jac_update_physical_steps=10,
        order_of_accuracy=1,
        turbulence_intensity_percent=100,
        N_crit=0.4,
        reconstruction_gradient_limiter=0.2,
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

    compare_to_ref(he, "../ref/case_params/heat_equation/ref.json", content_only=True)
