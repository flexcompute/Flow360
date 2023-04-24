import unittest

import pytest

import flow360 as fl

from .mock_server import mock_response
from .utils import compare_to_ref, mock_id, to_file_from_file_test

fl.UserConfig.disable_validation()
# fl.Flow360Params = fl.component.flow360_params.flow360_params.UnvalidatedFlow360Params

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_no_validation():
    params = fl.UnvalidatedFlow360Params("data/case_params/incorrect.json")
    compare_to_ref(params, "data/case_params/incorrect.json", content_only=True)
    to_file_from_file_test(params)
    print(params)


def test_no_validation_test_submit(mock_response):
    params = fl.UnvalidatedFlow360Params("data/case_params/incorrect.json")

    case = fl.Case.create(name="hi", params=params, volume_mesh_id=mock_id)
    case.submit()

    case_2 = case.copy()
    case_3 = case.retry()
    case_4 = case.fork()
    case_5 = case.continuation()
    print(case_5)
