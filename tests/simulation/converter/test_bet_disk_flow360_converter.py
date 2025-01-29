import json
import os
import re
from tempfile import NamedTemporaryFile

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.migration.BETDisk import (
    read_all_v1_BETDisks,
    read_single_v1_BETDisk,
)
from flow360.component.simulation.models.volume_models import BETDisk


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_single_flow360_bet_convert():
    disk = read_single_v1_BETDisk(
        file_path="./data/single_flow360_bet_disk.json",
        mesh_unit=1 * u.cm,
        time_unit=2 * u.s,
    )
    assert isinstance(disk, BETDisk)
    with open("./ref/ref_single_bet_disk.json", mode="r") as fp:
        ref_dict = json.load(fp=fp)
    compare_values(disk.model_dump(), ref_dict)

    with pytest.raises(
        ValueError,
        match=re.escape("Please pass in single BETDisk setting at a time."),
    ):
        # Wrong usage by supplying the complete Flow360.json
        disk = read_single_v1_BETDisk(
            file_path="./data/full_flow360.json",
            mesh_unit=1 * u.cm,
            time_unit=2 * u.s,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The supplied Flow360 input for BETDisk hsa invalid format. Details: 'axisOfRotation'."
        ),
    ):
        # Wrong usage by supplying incorrect schema json
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump({"adjfk": 1234}, temp_file)
            temp_file.flush()
            temp_file_name = temp_file.name

        try:
            read_single_v1_BETDisk(
                file_path=temp_file_name,
                mesh_unit=1 * u.cm,
                time_unit=2 * u.s,
            )
        finally:
            os.remove(temp_file_name)


def test_full_flow360_bet_convert():

    list_of_disks = read_all_v1_BETDisks(
        file_path="./data/full_flow360.json",
        mesh_unit=1 * u.cm,
        time_unit=2 * u.s,
    )

    assert isinstance(list_of_disks, list)
    assert len(list_of_disks) == 2
    assert all([isinstance(item, BETDisk) for item in list_of_disks])

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot find 'BETDisk' key in the supplied JSON file."),
    ):

        read_all_v1_BETDisks(
            file_path="./data/single_flow360_bet_disk.json",
            mesh_unit=1 * u.cm,
            time_unit=2 * u.s,
        )
