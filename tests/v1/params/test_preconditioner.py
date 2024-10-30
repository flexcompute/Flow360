import unittest

import numpy as np
import pydantic.v1 as pd
import pytest

import flow360 as fl
from flow360 import units as u
from flow360.component.v1.flow360_params import FreestreamFromVelocity

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_freesteam():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(low_mach_preconditioner=True),
        )

    case_params = params.to_solver()
    assert abs(case_params.navier_stokes_solver.low_mach_preconditioner_threshold - 0.84) < 0.001
