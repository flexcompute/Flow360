import json
import unittest

import numpy as np
import pydantic as pd
import pytest

import flow360 as fl
from flow360 import units as u
from flow360.component.flow360_params.flow360_params import (
    FreestreamFromMach,
    FreestreamFromVelocity,
    ZeroFreestream,
    ZeroFreestreamFromVelocity,
)
from flow360.exceptions import ConfigError
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_freesteam():
    with fl.SI_unit_system:
        params = fl.Flow360Params(fluid_properties=fl.air, geometry=fl.Geometry(mesh_unit=u.m))

        fs = FreestreamFromMach(Mach=1, temperature=300, mu_ref=1)
        assert fs

        to_file_from_file_test(fs)

        p = fl.Flow360Params(freestream={"Mach": 1, "temperature": 300, "mu_ref": 1})
        p = fl.Flow360Params(freestream={"Mach": 0, "Mach_ref": 1, "temperature": 300, "mu_ref": 1})
        p = fl.Flow360Params(freestream={"velocity": 1})

        with pytest.raises(pd.ValidationError):
            fs = FreestreamFromMach(Mach=-1, Temperature=100)

        velocity_meter_per_sec = 10
        fs = FreestreamFromVelocity(velocity=velocity_meter_per_sec * u.m / u.s)
        to_file_from_file_test(fs)
        assert fs

        fs_solver = fs.to_solver(params)
        ref_mach = velocity_meter_per_sec / np.sqrt(1.4 * 287.0529 * 288.15)
        assertions.assertAlmostEqual(fs_solver.Mach, ref_mach)

        with pytest.raises(pd.ValidationError):
            fs = FreestreamFromVelocity(velocity=0)

        fs = ZeroFreestreamFromVelocity(velocity=0, velocity_ref=velocity_meter_per_sec * u.m / u.s)
        to_file_from_file_test(fs)
        fs_solver = fs.to_solver(params)
        assertions.assertAlmostEqual(fs_solver.Mach_ref, ref_mach)
