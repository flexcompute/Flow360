import json
import unittest

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_params import (
    Freestream,
)
from flow360.exceptions import ConfigError

from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

def test_freesteam():
    fs = Freestream(Mach=1, temperature=300, density=1.22)
    assert fs
    with pytest.raises(ConfigError):
        print(fs.to_flow360_json())
    assert fs.to_flow360_json(mesh_unit_length=1)

    with pytest.raises(pd.ValidationError):
        fs = Freestream(Mach=-1, Temperature=100)

    fs = Freestream.from_speed(speed=(10, "m/s"))
    to_file_from_file_test(fs)
    assert fs

    fs = Freestream.from_speed(speed=10)
    assert fs

    with pytest.raises(ConfigError):
        print(fs.to_flow360_json())

    assert fs.to_flow360_json(mesh_unit_length=1)
    assert "speed" in json.loads(fs.json())
    assert "density" in json.loads(fs.json())
    assert "speed" not in json.loads(fs.to_flow360_json(mesh_unit_length=1))
    assert "density" not in json.loads(fs.to_flow360_json(mesh_unit_length=1))

    to_file_from_file_test(fs)