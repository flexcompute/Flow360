import json
import os
import re
from tempfile import NamedTemporaryFile

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import BETDisk, Flow360File


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_single_flow360_bet_convert():
    disk = BETDisk.from_flow360(
        file=Flow360File(file_name="./data/single_flow360_bet_disk.json"),
        mesh_unit=1 * u.cm,
        time_unit=2 * u.s,
    )
    assert isinstance(disk, BETDisk)

    with pytest.raises(
        ValueError,
        match=re.escape("Please pass in single BETDisk setting at a time."),
    ):
        # Wrong usage by supplying the complete Flow360.json
        disk = BETDisk.from_flow360(
            file=Flow360File(file_name="./data/full_flow360.json"),
            mesh_unit=1 * u.cm,
            time_unit=2 * u.s,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The supplied Flow360 input for BETDisk is invalid. Details: 'axisOfRotation'"
        ),
    ):
        # Wrong usage by supplying incorrect schema json
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump({"adjfk": 1234}, temp_file)
            temp_file.flush()
            temp_file_name = temp_file.name

        try:
            BETDisk.from_flow360(
                file=Flow360File(file_name=temp_file_name),
                mesh_unit=1 * u.cm,
                time_unit=2 * u.s,
            )
        finally:
            os.remove(temp_file_name)


def test_full_flow360_bet_convert():

    list_of_disks = BETDisk.read_flow360_BETDisk_list(
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

        BETDisk.read_flow360_BETDisk_list(
            file_path="./data/single_flow360_bet_disk.json",
            mesh_unit=1 * u.cm,
            time_unit=2 * u.s,
        )
