import unittest

import pytest

from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    ForcePerArea
)
from flow360.exceptions import ValidationError

from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

def test_actuator_disk():
    fpa = ForcePerArea(radius=[0, 1], thrust=[1, 1], circumferential=[1, 1])
    assert fpa
    ad = ActuatorDisk(center=(0, 0, 0), axis_thrust=(0, 0, 1), thickness=20, force_per_area=fpa)
    assert ad

    with pytest.raises(ValidationError):
        fpa = ForcePerArea(radius=[0, 1, 3], thrust=[1, 1], circumferential=[1, 1])

    to_file_from_file_test(ad)
    compare_to_ref(ad, "../ref/case_params/actuator_disk/json.json")
    compare_to_ref(ad, "../ref/case_params/actuator_disk/yaml.yaml")
