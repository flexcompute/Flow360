import unittest

import pytest

from flow360 import exceptions as ex
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder3D

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_draft_geometry_from_file():
    with pytest.raises(ex.Flow360FileError, match="Unsupported geometry file extensions"):
        sm = Geometry.from_file("file.unsupported")

    with pytest.raises(ex.Flow360FileError, match="not found"):
        sm = Geometry.from_file("data/geometry/no_exist.step")

    Cylinder3D.get_files()
    sm = Geometry.from_file(
        Cylinder3D.geometry,
        project_name="my_geo",
        solver_version="Mock_version-99.9.9",
        length_unit="cm",
    )
    assert sm.project_name == "my_geo"
    assert sm.length_unit == "cm"
