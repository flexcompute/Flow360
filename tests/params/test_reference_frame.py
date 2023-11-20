import json
import unittest

import numpy as np
import pydantic as pd
import pytest

import flow360 as fl
from flow360 import units as u
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    TimeStepping,
)
from flow360.exceptions import ConfigError, ValidationError
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_reference_frames():
    RPM = 10
    mesh_unit = 0.001

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1.0,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=fl.air,
            volume_zones={
                "zone1": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=RPM * u.rpm
                    )
                ),
                "zone2": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=RPM * 2 * fl.pi / 60
                    )
                ),
                "zone3": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=RPM * 360 / 60 * u.deg / u.s
                    )
                ),
            },
        )

    params_solver = params.to_solver()

    ref_C_inf = np.sqrt(1.4 * 287.0529 * 288.15)
    non_dim_omega = RPM * 2 * fl.pi / 60 / ref_C_inf * mesh_unit  # 3.0773317581937964e-06

    assertions.assertAlmostEqual(
        params_solver.volume_zones["zone1"].reference_frame.omega_radians.v.item(), non_dim_omega
    )

    assertions.assertAlmostEqual(
        params_solver.volume_zones["zone2"].reference_frame.omega_radians.v.item(), non_dim_omega
    )

    assertions.assertAlmostEqual(
        params_solver.volume_zones["zone3"].reference_frame.omega_radians.v.item(), non_dim_omega
    )
