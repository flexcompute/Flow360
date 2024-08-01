import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.simulation_params import SimulationParams
from tests.simulation.translator.utils.xv15_bet_disk_helper import createBETDiskSteady
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    _BET_cylinder,
    _rpm_hover_mode,
)

assertions = unittest.TestCase("__init__")


@pytest.fixture
def create_steady_bet_disk():
    bet_disk = createBETDiskSteady(_BET_cylinder, 10, _rpm_hover_mode)
    return bet_disk


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_initial_blade_direction(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    BETDisk.model_validate(bet_disk)

    with pytest.raises(
        ValueError,
        match="On one BET disk, the initial_blade_direction is required to specify since its blade_line_chord is non-zero",
    ):
        bet_disk_2 = bet_disk.model_copy(deep=True)
        bet_disk_2.blade_line_chord = 0.1 * u.inch


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_initial_blade_direction_with_bet_name(create_steady_bet_disk):
    with pytest.raises(
        ValueError,
        match="On the BET disk custom_bet_disk_name, the initial_blade_direction is required to specify since its blade_line_chord is non-zero",
    ):
        bet_disk = create_steady_bet_disk
        bet_disk.name = "custom_bet_disk_name"
        bet_disk.blade_line_chord = 0.1 * u.inch
