import unittest

import pydantic as pd
import pytest

import numpy as np

import flow360 as fl
from flow360.component.flow360_params.flow360_params import VolumeZones
from flow360.component.flow360_params.volume_zones import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    ReferenceFrame,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_volume_zones():
    with pytest.raises(pd.ValidationError):
        with fl.SI_unit_system:
            rf = ReferenceFrame(
                center=(0, 0, 0),
                axis=(0, 0, 1),
            )

    with fl.SI_unit_system:
        rf = ReferenceFrame(center=(0, 0, 0), axis=(0, 0, 1), omega=1)

    assert rf

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=0)

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source="0")

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=1)

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source="1")

    assert zone

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=-1)

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(), zone2=HeatTransferVolumeZone(thermal_conductivity=1)
    )

    assert zones

    with fl.SI_unit_system:
        param = fl.Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
            volume_zones={
                "zone1": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameDynamic(axis=(1, 2, 3), center=(1, 1, 1))
                ),
                "zone2": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameExpression(
                        axis=(1, 2, 3), center=(1, 1, 1), theta_degrees="0.2*t"
                    )
                ),
                "zone3": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameOmegaRadians(
                        axis=(1, 4, 3), center=(1, 1, 1), omega_radians=0.5
                    )
                ),
                "zone4": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameOmegaDegrees(
                        axis=(1, 5, 3), center=(1, 1, 1), omega_degrees=1.5
                    )
                ),
                "zone5": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(axis=(-5, 2, 3), center=(1, 1, 1), omega=2.5)
                ),
            },
        )

    param.flow360_json()
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone1"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone2"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone3"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone4"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone5"].reference_frame.axis)) - 1) < 1e-10
    )

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    to_file_from_file_test(zones)

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(reference_frame=rf),
        zone2=HeatTransferVolumeZone(
            thermal_conductivity=1,
            heat_capacity=1,
            initial_condition=InitialConditionHeatTransfer(T=100),
        ),
    )

    assert zones

    # to_file_from_file_test(zones)
