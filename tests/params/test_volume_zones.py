import unittest

import pydantic as pd
import pytest

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

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    to_file_from_file_test(zones)

    with fl.flow360_unit_system:
        zones = VolumeZones(
            zone1=FluidDynamicsVolumeZone(reference_frame=rf),
            zone2=HeatTransferVolumeZone(
                thermal_conductivity=1,
                heat_capacity=1,
                initial_condition=InitialConditionHeatTransfer(T=100),
            ),
        )

    assert zones
