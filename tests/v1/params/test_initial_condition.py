import unittest

import pytest

import flow360.component.v1xxx as fl
from flow360.component.v1.initial_condition import ExpressionInitialCondition
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_initial_condition():
    ic = ExpressionInitialCondition(rho="x*y", u="x+y", v="x-y", w="z+x+y", p="x/y")
    assert ic
    assert ic.type == "expression"

    to_file_from_file_test(ic)

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(meshUnit=1),
            boundaries={
                "MyBC": fl.FreestreamBoundary(),
            },
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(),
            initial_condition=fl.ExpressionInitialCondition(rho="1.0", u="2.0*y^2", v="2;"),
        )
        solver_params = params.to_solver()
        assert solver_params.initial_condition.rho == "1.0"
        assert solver_params.initial_condition.u == "2.0*powf(y, 2)"
        assert solver_params.initial_condition.v == "2;"
