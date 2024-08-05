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
def test_bet_disk_blade_line_chord(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match="On one BET disk, the blade_line_chord has to be positive since its initial_blade_direction is specified.",
    ):
        bet_disk.initial_blade_direction = (1, 0, 0)


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


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_disorder_alphas(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match="On one BET disk, the alphas are not in increasing order.",
    ):
        tmp = bet_disk.alphas[0]
        bet_disk.alphas[0] = bet_disk.alphas[1]
        bet_disk.alphas[1] = tmp
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_duplicate_chords(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match="On the BET disk diskABC, it has duplicated radius at 150.0348189415042 in chords.",
    ):
        bet_disk.name = "diskABC"
        bet_disk.chords.append(bet_disk.chords[-1])
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_duplicate_twists(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match="On the BET disk diskABC, it has duplicated radius at 150.0 in twists.",
    ):
        bet_disk.name = "diskABC"
        bet_disk.twists.append(bet_disk.twists[-1])
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_nonequal_sectional_radiuses_and_polars(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match=r"On the BET disk diskABC, the length of sectional_radiuses \(7\) is not the same as that of sectional_polars \(6\).",
    ):
        bet_disk.name = "diskABC"
        bet_disk.sectional_radiuses.append(bet_disk.sectional_radiuses[-1])
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_3d_coefficients_dimension_wrong_mach_numbers(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match=r"On the BET disk diskABC, \(cross section: 0\): number of mach_numbers = 2, but the first dimension of lift_coeffs is 1",
    ):
        bet_disk.name = "diskABC"
        bet_disk.mach_numbers.append(bet_disk.mach_numbers[-1])
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_3d_coefficients_dimension_wrong_re_numbers(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match=r"On the BET disk diskABC, \(cross section: 0\) \(Mach index \(0-based\) 0\): number of Reynolds = 2, but the second dimension of lift_coeffs is 1",
    ):
        bet_disk.name = "diskABC"
        bet_disk.reynolds_numbers.append(bet_disk.reynolds_numbers[-1])
        BETDisk.model_validate(bet_disk)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_3d_coefficients_dimension_wrong_alpha_numbers(create_steady_bet_disk):
    bet_disk = create_steady_bet_disk
    with pytest.raises(
        ValueError,
        match=r"On the BET disk diskABC, \(cross section: 0\) \(Mach index \(0-based\) 0, Reynolds index \(0-based\) 0\): number of Alphas = 18, but the third dimension of lift_coeffs is 17.",
    ):
        bet_disk.name = "diskABC"
        bet_disk.alphas.append(bet_disk.alphas[-1])
        BETDisk.model_validate(bet_disk)
