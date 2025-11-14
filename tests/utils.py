import difflib
import json
import os
import shutil
import tempfile
from numbers import Number

import numpy as np
import pytest
import unyt
from pylatex import Document

from flow360.cloud.rest_api import RestApi
from flow360.cloud.s3_utils import S3TransferType, get_local_filename_and_create_folders
from flow360.plugins.report import report_doc

here = os.path.dirname(os.path.abspath(__file__))


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


def compare_dict_to_ref(data, ref_path):
    with open(ref_path) as fh:
        ref_data = json.load(fh)
    equal = sorted(data.items()) == sorted(ref_data.items())
    if equal is False:
        show_dict_diff(data, ref_data)
        assert equal


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
            compare_dict_to_ref(a, ref_path)


@pytest.fixture()
def s3_download_override(monkeypatch):
    def s3_mock_download(
        resource_id: str,
        remote_file_name: str,
        to_file: str = None,
        to_folder: str = ".",
        overwrite: bool = True,
        progress_callback=None,
        log_error=True,
        **kwargs,
    ):
        from botocore.exceptions import ClientError as CloudFileNotFoundError

        full_remote_path = os.path.join(here, "data", resource_id, remote_file_name)
        print(f"DEBUG: looking for {remote_file_name=} at {full_remote_path=}")
        if not os.path.exists(full_remote_path):
            raise CloudFileNotFoundError(
                error_response={"Error": {"Message": f"file not found: {remote_file_name}"}},
                operation_name="download",
            )
        to_file = get_local_filename_and_create_folders(
            remote_file_name, to_file=to_file, to_folder=to_folder
        )
        abs_src = os.path.abspath(full_remote_path)
        to_file = os.path.abspath(to_file)
        if abs_src != to_file:
            kwargs.pop("verbose", None)
            shutil.copy(abs_src, to_file, **kwargs)
            print(f"MOCK_DOWNLOAD: Saved to {to_file}")

    monkeypatch.setattr(S3TransferType.CASE, "download_file", s3_mock_download)
    monkeypatch.setattr(S3TransferType.VOLUME_MESH, "download_file", s3_mock_download)
    monkeypatch.setattr(S3TransferType.SURFACE_MESH, "download_file", s3_mock_download)
    monkeypatch.setattr(S3TransferType.GEOMETRY, "download_file", s3_mock_download)


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


@pytest.fixture()
def generate_pdf(monkeypatch):
    def mock_generate_pdf(*args, **kwargs):
        print("MOCK generate_pdf!!!")

    monkeypatch.setattr(Document, "generate_pdf", mock_generate_pdf)


@pytest.fixture
def mock_detect_latex_compiler(monkeypatch):
    def _mock(*args, **kwargs):
        print("MOCK detect_latex_compiler called!")
        return ("xelatex", [])

    monkeypatch.setattr(report_doc, "detect_latex_compiler", _mock)
