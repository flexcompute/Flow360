import unittest

import pytest

import flow360.component.v1xxx as fl
from flow360.component.validator import Validator

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_():
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=1.15315084119231,
            moment_length=(1.47602, 0.801672958512342, 1.47602),
            mesh_unit="m",
        ),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        time_stepping=fl.UnsteadyTimeStepping(max_pseudo_steps=500),
        boundaries={
            "1": fl.NoSlipWall(name="wing"),
            "2": fl.SlipWall(name="symmetry"),
            "3": fl.FreestreamBoundary(name="freestream"),
        },
    )

    Validator.CASE.validate(params, mesh_id="0000-0000-0000-0000")
