import unittest

import pytest

from flow360.component.flow360_params.flow360_params import (
    ExpressionInitialCondition,
    FreestreamInitialCondition,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_initial_condition():
    ic = FreestreamInitialCondition()
    assert ic
    assert ic.model_type == "freestream"

    ic = ExpressionInitialCondition(rho="x*y", u="x+y", v="x-y", w="z+x+y", p="x/y")
    assert ic
    assert ic.model_type == "expression"

    to_file_from_file_test(ic)
