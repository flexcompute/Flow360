import os

import pytest

from flow360.examples import Airplane, Cylinder2D, OM6wing, base_test_case
from flow360.examples.base_test_case import BaseTestCase


def test_om6_example_file_download():
    try:
        os.remove(OM6wing.mesh_filename)
    except FileNotFoundError:
        pass

    with pytest.raises(FileNotFoundError):
        print(OM6wing.mesh_filename)

    OM6wing.get_files()
    assert os.path.exists(OM6wing.case_json)
    assert os.path.exists(OM6wing.case_yaml)
    assert os.path.exists(OM6wing.mesh_json)
    assert os.path.exists(OM6wing.mesh_filename)


def test_om6_release_22_3_3_0_example():
    OM6wing.set_version("release-22.3.2.0")
    OM6wing.get_files()
    assert os.path.exists(OM6wing.case_json)
    assert os.path.exists(OM6wing.mesh_json)
    assert os.path.exists(OM6wing.mesh_filename)


def test_cylinder_example():
    Cylinder2D.get_files()
    assert os.path.exists(Cylinder2D.case_json)
    assert os.path.exists(Cylinder2D.mesh_json)
    assert os.path.exists(Cylinder2D.mesh_filename)


def test_airplane_example():
    Airplane.get_files()
    assert os.path.exists(Airplane.geometry)
    assert os.path.exists(Airplane.surface_json)
    assert os.path.exists(Airplane.volume_json)


def test_version_parse():
    class TestCase(BaseTestCase):
        name = "data/examples"

        class url:
            case_json = "local://flow360.json"

    base_test_case.here = os.path.dirname(os.path.abspath(__file__))

    TestCase.set_version("release-23.1.3.0")
    assert TestCase._get_version_prefix() == "release-23.1.3.0"

    TestCase.set_version("release-23.1.2.0")
    assert TestCase._get_version_prefix() == "release-22.3.3.0lt"

    TestCase.set_version("release-22.3.3.0")
    assert TestCase._get_version_prefix() == "release-22.2.3.0le"

    TestCase.set_version("release-22.3.2.0")
    assert TestCase._get_version_prefix() == "release-22.2.3.0le"

    TestCase.set_version("release-22.2.3.0")
    assert TestCase._get_version_prefix() == "release-22.2.3.0le"

    TestCase.set_version("release-22.1.3.2")
    assert TestCase._get_version_prefix() == "release-22.1.3.2"

    TestCase.set_version("release-22.1.3.1")
    assert TestCase._get_version_prefix() == ""

    TestCase.set_version("release-22.1.3.0")
    assert TestCase._get_version_prefix() == "release-22.1.3.0ge"

    TestCase.set_version("release-22.1.2.9")
    assert TestCase._get_version_prefix() == "release-22.1.3.0ge"

    TestCase.set_version("release-22.1.2.0")
    assert TestCase._get_version_prefix() == "release-22.1.3.0ge"

    TestCase.set_version("release-22.1.1.0")
    assert TestCase._get_version_prefix() == "release-22.1.2.0gt"
