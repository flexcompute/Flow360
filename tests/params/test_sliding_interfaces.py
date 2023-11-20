import json
import math
import unittest

import pytest

from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    Freestream,
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
        omega=1,
    )

    msi = MeshSlidingInterface.from_case_sliding_interface(si)
    assert msi


def test_sliding_interface():
    with pytest.raises(ConfigError):
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
        omega=1,
        rpm=1,
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
        omega=1,
        rpm=1,
    )

    assert si
    to_file_from_file_test(si)

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=[0, 1],
        omegaRadians=1,
        rpm=1,
    )

    assert si

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(1, "rad/s"),
    )

    assert si
    assert si.json()

    to_file_from_file_test(si)
    compare_to_ref(si, "../ref/case_params/sliding_interface/json.json")
    compare_to_ref(si, "../ref/case_params/sliding_interface/yaml.yaml")

    with pytest.raises(ConfigError):
        print(si.to_flow360_json())

    assert "omega" in json.loads(si.json())
    assert "omegaRadians" not in json.loads(si.json())
    assert "omega" not in json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))
    assert "omegaRadians" in json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))
    assert json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))["omegaRadians"] == 0.01

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(1, "deg/s"),
    )

    assert si
    assert si.json()

    with pytest.raises(ConfigError):
        print(si.to_flow360_json())

    assert "omega" in json.loads(si.json())
    assert "omegaDegrees" not in json.loads(si.json())
    assert "omega" not in json.loads(si.to_flow360_json(mesh_unit_length=1, C_inf=1))
    assert "omegaDegrees" in json.loads(si.to_flow360_json(mesh_unit_length=1, C_inf=1))
    assert json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))["omegaDegrees"] == 0.01

    rpm = 100
    si_rpm = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        rpm=rpm,
    )

    si_omega = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(rpm * 2 * math.pi / 60, "rad/s"),
    )

    assert si_rpm.to_flow360_json(mesh_unit_length=0.01, C_inf=1) == si_omega.to_flow360_json(
        mesh_unit_length=0.01, C_inf=1
    )

    params = Flow360Params(sliding_interfaces=[si], freestream=Freestream.from_speed(10))

    assert params.json()
    with pytest.raises(ConfigError):
        print(params.to_flow360_json())

    params = Flow360Params(
        geometry=Geometry(mesh_unit="mm"),
        freestream=Freestream.from_speed(10),
        sliding_interfaces=[si],
    )

    assert params.json()
    assert params.to_flow360_json()
    assertions.assertAlmostEqual(
        json.loads(params.to_flow360_json())["slidingInterfaces"][0]["omegaDegrees"], 2.938e-06
    )
