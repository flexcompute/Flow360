import unittest

import pytest

from flow360.component.flow360_params.initial_condition import (
    ExpressionInitialCondition,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_initial_condition():
    ic = ExpressionInitialCondition(
        constants={"alpha": "0.01"}, rho="x*y", u="x+y", v="x-y", w="z+x+y", p="x/y"
    )
    assert ic

    to_file_from_file_test(ic)
