import unittest

import pytest

from flow360 import exceptions as ex
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_draft_geometry_from_file():
    with pytest.raises(ex.Flow360FileError, match="Unsupported geometry file extensions"):
        sm = Geometry.from_file("file.unsupported")

    with pytest.raises(ex.Flow360FileError, match="not found"):
        sm = Geometry.from_file("data/geometry/no_exist.step")

    Cylinder.get_files()
    sm = Geometry.from_file(Cylinder.geometry)
    sm = Geometry.from_file(Cylinder.geometry)
    assert sm
