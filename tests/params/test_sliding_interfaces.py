import json
import math
import unittest

import pytest

from flow360 import units as u
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    MeshSlidingInterface,
    SlidingInterface,
)
from flow360.exceptions import ConfigError
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_mesh_sliding_interface():
    msi = MeshSlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
    )
    assert msi
    to_file_from_file_test(msi)

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega_radians=1,
    )

    msi = MeshSlidingInterface.from_case_sliding_interface(si)
    assert msi


def test_sliding_interface():
    with pytest.raises(ValueError):
        si = SlidingInterface(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            stationary_patches=["patch1"],
            rotating_patches=["patch2"],
            volume_name="volume1",
        )

    # setting up both omega and rpm, or
    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=1,
        omega_radians=1,
    )

    assert si

    si = SlidingInterface.parse_raw(
        """
    {
        "stationaryPatches" : ["farField/rotationInterface"],
        "rotatingPatches" : ["innerRotating/rotationInterface"],
        "axisOfRotation" : [0,0,-1],
        "centerOfRotation" : [0,0,0],
        "omegaRadians" : 1.84691e-01,
        "volumeName" : ["innerRotating"]
    }
    """
    )

    assert si

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=["volume1", "volume2"],
        omega_radians=1,
    )

    assert si
    to_file_from_file_test(si)

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omegaRadians=1,
    )

    assert si
    assert si.json()

    to_file_from_file_test(si)
    compare_to_ref(si, "../ref/case_params/sliding_interface/json.json")
    compare_to_ref(si, "../ref/case_params/sliding_interface/yaml.yaml")

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega_degrees=1,
    )

    assert si
    assert si.json()
