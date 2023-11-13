import unittest

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_params import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    ReferenceFrame,
    VolumeZones
)
from flow360.exceptions import ConfigError

from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_volume_zones():
    with pytest.raises(ConfigError):
        rf = ReferenceFrame(
            center=(0, 0, 0),
            axis=(0, 0, 1),
        )

    rf = ReferenceFrame(center=(0, 0, 0), axis=(0, 0, 1), omega_radians=1)

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

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(reference_frame=rf),
        zone2=HeatTransferVolumeZone(
            thermal_conductivity=1,
            heat_capacity=1,
            initial_condition=InitialConditionHeatTransfer(T_solid=100),
        ),
    )

    assert zones

    to_file_from_file_test(zones)
