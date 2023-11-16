import unittest

import pytest

from flow360.component.flow360_params.flow360_temp import (
    PorousMedium,
    PorousMediumVolumeZone,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_porous_media():
    pmvz = PorousMediumVolumeZone(
        zone_type="box",
        center=[1, 2, 3],
        lengths=[3, 4, 5],
        axes=[[0, 1, 0], [1, 0, 0]],
        windowing_lengths=[0.5, 0.5, 0.5],
    )

    assert pmvz

    pm = PorousMedium(
        darcy_coefficient=[1, 1, 1], forchheimer_coefficient=[1, 1, 1], volume_zone=pmvz
    )

    assert pm

    to_file_from_file_test(pm)
