import difflib
import json
import os
import tempfile

from flow360.cloud.rest_api import RestApi

mock_id = "00000000-0000-0000-0000-000000000000"


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
            assert sorted(a.items()) == sorted(b.items())


# for generating MOCK WEBAPI data:


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
