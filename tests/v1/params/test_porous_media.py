import unittest

import pytest

from flow360 import SI_unit_system
from flow360.component.v1.flow360_params import PorousMediumBox
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.usefixtures("array_equality_override")
def test_porous_media():
    with SI_unit_system:
        pm = PorousMediumBox(
            darcy_coefficient=[1, 1, 1],
            forchheimer_coefficient=[1, 1, 1],
            center=[1, 2, 3],
            lengths=[3, 4, 5],
            axes=[[0, 1, 0], [1, 0, 0]],
            windowing_lengths=[0.5, 0.5, 0.5],
        )

    assert pm

    to_file_from_file_test(pm)
