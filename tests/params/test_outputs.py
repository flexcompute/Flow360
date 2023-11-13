import unittest

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_params import (
    AeroacousticOutput,
)

from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

def test_aeroacoustic_output():
    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput()

    output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)])

    to_file_from_file_test(output)

    assert output

    output = AeroacousticOutput(observers=[])

    to_file_from_file_test(output)

    assert output

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=0)

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=-2
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=2
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=0
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=-1, animation_frequency_offset=0
    )

    assert output

    to_file_from_file_test(output)

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(
            observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=-2, animation_frequency_offset=0
        )

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)],
        animation_frequency=1,
        animation_frequency_offset=-2,
        patch_type="solid",
    )

    assert output

    to_file_from_file_test(output)

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(
            observers=[(0, 0, 0), (0, 1, 1)],
            animation_frequency=1,
            animation_frequency_offset=-2,
            patch_type="other",
        )