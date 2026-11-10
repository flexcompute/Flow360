"""Tests for the service-level BET Disk translators consumed by the webservice."""

import json
import os

import pytest

from flow360.component.simulation.services import translate_charm_bet_disk

CHARM_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "charm")


def _read(file_name: str) -> str:
    with open(os.path.join(CHARM_DATA_DIR, file_name), mode="r", encoding="utf-8") as fp:
        return fp.read()


@pytest.fixture(name="charm_files")
def fixture_charm_files():
    return _read("genericbg.inp"), {"0012.inp": _read("0012.inp")}


def test_translate_charm_bet_disk(charm_files):
    blade_geometry, airfoil_tables = charm_files
    bet_dict_list, errors = translate_charm_bet_disk(
        geometry_file_content=blade_geometry,
        polar_file_contents_dict=airfoil_tables,
        length_unit="inch",
        angle_unit="deg",
    )
    assert errors == []
    assert len(bet_dict_list) == 1

    bet_dict = bet_dict_list[0]
    assert set(bet_dict.keys()) == {
        "sectional_radiuses",
        "twists",
        "chords",
        "mach_numbers",
        "reynolds_numbers",
        "alphas",
        "sectional_polars",
    }
    # Units are serialized out of unyt into the {"value", "units"} wire form.
    assert bet_dict["sectional_radiuses"] == [
        {"value": 3.0, "units": "inch"},
        {"value": 30.0, "units": "inch"},
    ]
    assert bet_dict["chords"] == [
        {"radius": {"value": 3.0, "units": "inch"}, "chord": {"value": 2.0, "units": "inch"}},
        {"radius": {"value": 30.0, "units": "inch"}, "chord": {"value": 2.0, "units": "inch"}},
    ]
    # Root TWRD=6.5 plus the per-segment TWSTGD=-9.0. The fixture writes that
    # keyword as "TWSTGD(ISEG" with no closing paren, so this also pins the
    # keyword parser's tolerance for CHARM's loose parenthetical suffixes.
    assert bet_dict["twists"] == [
        {"radius": {"value": 3.0, "units": "inch"}, "twist": {"value": 6.5, "units": "degree"}},
        {"radius": {"value": 30.0, "units": "inch"}, "twist": {"value": -2.5, "units": "degree"}},
    ]
    assert len(bet_dict["sectional_polars"]) == len(bet_dict["sectional_radiuses"])
    # The whole response is JSON-serializable; the webservice dumps it with json.dumps.
    json.dumps(bet_dict)


@pytest.mark.parametrize("polar_file_contents_dict", [{}, {"a.inp": "x", "b.inp": "y"}])
def test_translate_charm_bet_disk_requires_exactly_one_airfoil_tables_file(
    charm_files, polar_file_contents_dict
):
    blade_geometry, _ = charm_files
    bet_dict_list, errors = translate_charm_bet_disk(
        geometry_file_content=blade_geometry,
        polar_file_contents_dict=polar_file_contents_dict,
        length_unit="inch",
        angle_unit="deg",
    )
    assert bet_dict_list == []
    assert len(errors) == 1
    assert "Exactly one CHARM airfoil tables file is expected" in errors[0]


def test_translate_charm_bet_disk_reports_malformed_geometry(charm_files):
    _, airfoil_tables = charm_files
    bet_dict_list, errors = translate_charm_bet_disk(
        geometry_file_content="not a CHARM blade geometry file",
        polar_file_contents_dict=airfoil_tables,
        length_unit="inch",
        angle_unit="deg",
    )
    assert bet_dict_list == []
    assert len(errors) == 1
