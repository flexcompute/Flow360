import difflib
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import unyt

from flow360.cloud.rest_api import RestApi
from flow360.cloud.s3_utils import S3TransferType, get_local_filename_and_create_folders
from flow360.component.flow360_params import unit_system


@pytest.fixture
def mock_id():
    return "00000000-0000-0000-0000-000000000000"


def file_compare(file1, file2):
    with open(file1) as fh1:
        with open(file2) as fh2:
            diff = difflib.unified_diff(
                fh1.readlines(),
                fh2.readlines(),
                fromfile=file1,
                tofile=file2,
            )
            different = False
            for line in diff:
                print(line)
                different = True

            return not different


def show_dict_diff(dict1, dict2):
    def dict_to_sorted_lines(d):
        sorted_lines = []
        for key, value in sorted(d.items(), key=lambda item: item[0]):
            if isinstance(value, dict):
                # Recursively sort nested dictionaries
                value = "{" + ", ".join(dict_to_sorted_lines(value)) + "}"
            sorted_lines.append(f"{key}: {value}")
        return sorted_lines

    dict1_lines = dict_to_sorted_lines(dict1)
    dict2_lines = dict_to_sorted_lines(dict2)

    # Generate the diff
    diff = difflib.unified_diff(dict1_lines, dict2_lines, fromfile="dict1", tofile="dict2")
    # Printing the diff
    print("diff")
    print("\n".join(diff))
    print("end of diff")


def compare_dicts(dict1, dict2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()

    # Filter out the keys to be ignored
    dict1_filtered = {k: v for k, v in dict1.items() if k not in ignore_keys}
    dict2_filtered = {k: v for k, v in dict2.items() if k not in ignore_keys}

    if dict1_filtered.keys() != dict2_filtered.keys():
        print(f"dict keys not equal, dict1 {dict1_filtered.keys()}, dict2 {dict2_filtered.keys()}")
        return False

    for key in dict1_filtered:
        value1 = dict1_filtered[key]
        value2 = dict2_filtered[key]

        if not compare_values(value1, value2, atol, rtol, ignore_keys):
            print(f"dict value of key {key} not equal dict1 {dict1[key]}, dict2 {dict2[key]}")
            return False

    return True


def compare_values(value1, value2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    if isinstance(value1, float) and isinstance(value2, float):
        return np.isclose(value1, value2, rtol, atol)
    elif isinstance(value1, dict) and isinstance(value2, dict):
        return compare_dicts(value1, value2, atol, rtol, ignore_keys)
    elif isinstance(value1, list) and isinstance(value2, list):
        return compare_lists(value1, value2, atol, rtol, ignore_keys)
    else:
        return value1 == value2


def compare_lists(list1, list2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    if len(list1) != len(list2):
        return False

    if list1 and not isinstance(list1[0], dict):
        list1, list2 = sorted(list1), sorted(list2)

    for item1, item2 in zip(list1, list2):
        if not compare_values(item1, item2, atol, rtol, ignore_keys):
            print(f"list value not equal list1 {item1}, list2 {item2}")
            return False

    return True


def to_file_from_file_test(obj):
    test_extentions = ["yaml", "json"]
    factory = obj.__class__
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in test_extentions:
            obj_filename = os.path.join(tmpdir, f"obj.{ext}")
            obj.to_file(obj_filename)
            obj_read = factory.from_file(obj_filename)
            assert obj == obj_read
            obj_read = factory(filename=obj_filename)
            assert obj == obj_read


def compare_to_ref(obj, ref_path, content_only=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, f"file{os.path.splitext(ref_path)[1]}")
        obj.to_file(filename)

        if not content_only:
            assert file_compare(filename, ref_path)
        else:
            assert os.path.splitext(ref_path)[1] == ".json"
            with open(filename) as fh:
                a = json.load(fh)
            with open(ref_path) as fh:
                b = json.load(fh)
            equal = sorted(a.items()) == sorted(b.items())
            if equal is False:
                show_dict_diff(a, b)
                assert equal


@pytest.fixture()
def array_equality_override():
    # Save original methods
    original_unyt_eq = unyt.unyt_array.__eq__
    original_unyt_ne = unyt.unyt_array.__ne__
    original_flow360_eq = unit_system._Flow360BaseUnit.__eq__
    original_flow360_ne = unit_system._Flow360BaseUnit.__ne__

    # Overload equality for unyt arrays
    def unyt_array_eq(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(other, unit_system._Flow360BaseUnit):
            return flow360_unit_array_eq(other, self)
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__eq__(self, other)
        elif self.size == other.size:
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def unyt_array_ne(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(other, unit_system._Flow360BaseUnit):
            return flow360_unit_array_ne(other, self)
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__ne__(self, other)
        elif self.size == other.size:
            return any(self[i] != other[i] for i in range(len(self)))
        return True

    def flow360_unit_array_eq(
        self: unit_system._Flow360BaseUnit, other: unit_system._Flow360BaseUnit
    ):
        if isinstance(other, (unit_system._Flow360BaseUnit, unyt.unyt_array)):
            if self.size == other.size:
                if str(self.units) == str(other.units):
                    if self.size == 1:
                        return np.ndarray.__eq__(self.v, other.v)
                    if isinstance(other, unyt.unyt_array):
                        other = unit_system._Flow360BaseUnit.factory(other.v, str(other.units))
                    return all(np.ndarray.__eq__(v.v, o.v) for v, o in zip(self, other))
        return False

    def flow360_unit_array_ne(
        self: unit_system._Flow360BaseUnit, other: unit_system._Flow360BaseUnit
    ):
        if isinstance(other, (unit_system._Flow360BaseUnit, unyt.unyt_array)):
            if self.size == other.size:
                if str(self.units) == str(other.units):
                    if self.size == 1:
                        return np.ndarray.__ne__(self.v, other.v)
                    if isinstance(other, unyt.unyt_array):
                        other = unit_system._Flow360BaseUnit.factory(other.v, str(other.units))
                    return any(np.ndarray.__ne__(v.v, o.v) for v, o in zip(self, other))
        return True

    unyt.unyt_array.__eq__ = unyt_array_eq
    unyt.unyt_array.__ne__ = unyt_array_ne
    unit_system._Flow360BaseUnit.__eq__ = flow360_unit_array_eq
    unit_system._Flow360BaseUnit.__ne__ = flow360_unit_array_ne

    # Yield control to the test
    yield

    # Restore original methods
    unyt.unyt_array.__eq__ = original_unyt_eq
    unyt.unyt_array.__ne__ = original_unyt_ne
    unit_system._Flow360BaseUnit.__eq__ = original_flow360_eq
    unit_system._Flow360BaseUnit.__ne__ = original_flow360_ne


@pytest.fixture()
def s3_download_override():
    def s3_mock_download(
        resource_id: str,
        remote_file_name: str,
        to_file: str = None,
        to_folder: str = ".",
        overwrite: bool = True,
        progress_callback=None,
        log_error=True,
    ):
        to_file = get_local_filename_and_create_folders(
            remote_file_name, to_file=to_file, to_folder=to_folder
        )
        shutil.copy(os.path.join("data", remote_file_name), to_file)
        print(f"MOCK_DOWNLOAD: Saved to {to_file}")

    S3TransferType.CASE.download_file = s3_mock_download


# for generating MOCK WEBAPI data:
def generate_mock_webapi_data_version_check():
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join(current_directory, "data", "mock_webapi", "versions.json")
    with open(relative_path, "w") as fh:
        json.dump({"data": [{"version": "1.0.0"}, {"version": "2.0.3b5"}]}, fh, indent=4)


def empty_mock_webapi_data_version_check():
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join(current_directory, "data", "mock_webapi", "versions.json")
    with open(relative_path, "w") as fh:
        json.dump({"data": []}, fh, indent=4)


def generate_mock_webapi_data_volumemesh():
    resp = RestApi(endpoint="volumemeshes/page").get(
        params={"includeDeleted": True, "limit": 100, "start": 0}
    )
    print(resp)

    with open("volumemesh_pages_webapi_resp.json", "w") as fh:
        json.dump({"data": resp}, fh, indent=4)


def generate_mock_webapi_data_one_volumemesh():
    id = "6504db70-0edc-4eb0-ab26-5d90c1e2a049"
    resp = RestApi(endpoint=f"volumemeshes/{id}").get()
    print(resp)

    with open("volume_mesh_meta.json", "w") as fh:
        json.dump({"data": resp}, fh, indent=4)


def generate_mock_webapi_data_one_case():
    id = "c58e7a75-e349-476a-9020-247af6b2e92b"
    resp = RestApi(endpoint=f"cases/{id}").get()
    print(resp)

    with open("case_meta.json", "w") as fh:
        json.dump({"data": resp}, fh, indent=4)


def generate_mock_webapi_data_one_case_params():
    id = "c58e7a75-e349-476a-9020-247af6b2e92b"
    resp = RestApi(endpoint=f"cases/{id}/runtimeParams").get()
    print(resp)

    with open("case_params.json", "w") as fh:
        json.dump({"data": resp}, fh, indent=4)
