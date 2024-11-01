import unittest

import numpy as np
import pydantic.v1 as pd
import pytest

import flow360.v1 as fl
from flow360.component.v1 import units as u
from flow360.component.v1.flow360_params import (
    FreestreamFromMach,
    FreestreamFromMachReynolds,
    FreestreamFromVelocity,
    ZeroFreestream,
    ZeroFreestreamFromVelocity,
)
from tests.utils import to_file_from_file_test

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
            freestream=FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
        )

        fs = FreestreamFromMach(Mach=1, temperature=300, mu_ref=1)
        assert fs

        to_file_from_file_test(fs)

        fl.Flow360Params(
            freestream={"modelType": "FromMach", "Mach": 1, "temperature": 300, "mu_ref": 1},
            boundaries={},
        )
        fl.Flow360Params(
            freestream={
                "modelType": "ZeroMach",
                "Mach": 0,
                "Mach_ref": 1,
                "temperature": 300,
                "mu_ref": 1,
            },
            boundaries={},
        )
        fl.Flow360Params(
            freestream={"modelType": "FromVelocity", "velocity": 1},
            boundaries={},
        )

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
            fs = FreestreamFromVelocity(velocity=0 * u.m / u.s)

        fs = ZeroFreestreamFromVelocity(velocity=0, velocity_ref=velocity_meter_per_sec * u.m / u.s)
        to_file_from_file_test(fs)
        fs_solver = fs.to_solver(params)
        assertions.assertAlmostEqual(fs_solver.Mach_ref, ref_mach)

        fs = FreestreamFromMachReynolds(Mach=0.1, Reynolds="inf", temperature=288.15)
        to_file_from_file_test(fs)

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=FreestreamFromMach(
                Mach=1, temperature=288.15, mu_ref=1, turbulent_viscosity_ratio=0.001
            ),
        )
    case_params = params.to_solver()
    assert case_params.freestream.turbulence_quantities.turbulent_viscosity_ratio == 0.001

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=FreestreamFromMachReynolds(
                Mach=0.1, Reynolds="inf", temperature=288.15, turbulent_viscosity_ratio=0.002
            ),
        )
    case_params = params.to_solver()
    assert case_params.freestream.turbulence_quantities.turbulent_viscosity_ratio == 0.002

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=FreestreamFromVelocity(
                velocity=5 * u.m / u.s, turbulent_viscosity_ratio=0.003
            ),
        )
    case_params = params.to_solver()
    assert case_params.freestream.turbulence_quantities.turbulent_viscosity_ratio == 0.003

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=ZeroFreestream(
                Mach_ref=0.123, mu_ref=1e-5, temperature=288.15, turbulent_viscosity_ratio=0.004
            ),
        )
    case_params = params.to_solver()
    assert case_params.freestream.turbulence_quantities.turbulent_viscosity_ratio == 0.004

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=u.m),
            boundaries={},
            freestream=ZeroFreestreamFromVelocity(
                velocity=0, velocity_ref=1 * u.m / u.s, turbulent_viscosity_ratio=0.005
            ),
        )
    case_params = params.to_solver()
    assert case_params.freestream.turbulence_quantities.turbulent_viscosity_ratio == 0.005
