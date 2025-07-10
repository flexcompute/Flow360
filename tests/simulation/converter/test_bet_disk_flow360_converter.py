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


def test_single_flow360_bet_convert(atol=1e-15, rtol=1e-10, debug=False):
    disk = read_single_v1_BETDisk(
        file_path="./data/single_flow360_bet_disk.json",
        mesh_unit=1 * u.cm,
        freestream_temperature=288.15 * u.K,
        bet_disk_name="MyBETDisk",
    )
    assert isinstance(disk, BETDisk)
    disk = disk.model_dump_json()
    disk = json.loads(disk)
    del disk["entities"]["stored_entities"][0]["private_attribute_id"]
    del disk["private_attribute_input_cache"]["entities"]["stored_entities"][0][
        "private_attribute_id"
    ]
    with open("./ref/ref_single_bet_disk.json", mode="r") as fp:
        ref_dict = json.load(fp=fp)
    if debug:
        print(">>> disk = ", disk)
        print("=== disk ===\n", json.dumps(disk, indent=4, sort_keys=True))
        print("=== ref_dict ===\n", json.dumps(ref_dict, indent=4, sort_keys=True))
    assert compare_values(ref_dict, disk, atol=atol, rtol=rtol)

    with pytest.raises(
        ValueError,
        match=re.escape("Please pass in single BETDisk setting at a time."),
    ):
        # Wrong usage by supplying the complete Flow360.json
        disk = read_single_v1_BETDisk(
            file_path="./data/full_flow360.json",
            mesh_unit=1 * u.cm,
            freestream_temperature=288.15 * u.K,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The supplied Flow360 input for BETDisk has invalid format. Details: 'axisOfRotation'."
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
                freestream_temperature=288.15 * u.K,
            )
        finally:
            os.remove(temp_file_name)


def test_full_flow360_bet_convert():

    list_of_disks = read_all_v1_BETDisks(
        file_path="./data/full_flow360.json",
        mesh_unit=1 * u.cm,
        freestream_temperature=288.15 * u.K,
    )

    assert isinstance(list_of_disks, list)
    assert len(list_of_disks) == 2
    assert list_of_disks[0].name == "Disk0"
    assert list_of_disks[1].name == "Disk1"
    assert list_of_disks[0].entities.stored_entities[0].name == "bet_cylinder_0"
    assert list_of_disks[1].entities.stored_entities[0].name == "bet_cylinder_1"
    assert all([isinstance(item, BETDisk) for item in list_of_disks])

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot find 'BETDisk' key in the supplied JSON file."),
    ):

        read_all_v1_BETDisks(
            file_path="./data/single_flow360_bet_disk.json",
            mesh_unit=1 * u.cm,
            freestream_temperature=288.15 * u.K,
        )
