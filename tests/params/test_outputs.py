import unittest

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_output import (
    IsoSurface,
    IsoSurfaceOutput,
    MonitorOutput,
    ProbeMonitor,
    SliceOutput,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.flow360_params.flow360_params import AeroacousticOutput
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


def test_surface_output():
    output = SurfaceOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            animation_frequency=-2,
            animation_frequency_time_average=-1,
            output_fields=["Cp", "qcriterion"],
        )

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            animation_frequency=-1,
            animation_frequency_time_average=-2,
            output_fields=["Cp", "qcriterion"],
        )

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            animation_frequency=-1,
            animation_frequency_time_average=-1,
            output_fields=["invalid_field", "qcriterion"],
        )

    output = SurfaceOutput(
        animation_frequency=-1,
        animation_frequency_time_average=-1,
        output_fields=["Cp", "qcriterion"],
    )

    assert output

    to_file_from_file_test(output)


def test_slice_output():
    output = SliceOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(animation_frequency=-2, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(animation_frequency=-1, output_fields=["invalid_field", "qcriterion"])

    output = SliceOutput(animation_frequency=-1, output_fields=["Cp", "qcriterion"])

    assert output

    to_file_from_file_test(output)


def test_volume_output():
    output = VolumeOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=-2, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=-1, output_fields=["invalid_field", "qcriterion"])

    output = VolumeOutput(animation_frequency=-1, output_fields=["Cp", "qcriterion"])

    assert output

    to_file_from_file_test(output)


def test_iso_surface_output():
    iso_surface = IsoSurface(
        surface_field_magnitude=0.5,
        surface_field=[
            "p",
            "Mach",
            "qcriterion",
        ],
        output_fields=["Cp", "qcriterion"],
    )

    assert iso_surface

    output = IsoSurfaceOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = IsoSurfaceOutput(
            animation_frequency=-2,
            iso_surfaces={"s1": iso_surface},
        )

    output = IsoSurfaceOutput(
        animation_frequency=-1,
        iso_surfaces={"s1": iso_surface},
    )

    assert output

    to_file_from_file_test(output)


def test_monitor_output():
    probe = ProbeMonitor(
        monitor_locations=[[0, 0, 0], [0, 10, 0.4]], output_fields=["Cp", "qcriterion"]
    )

    assert probe

    integral = SurfaceIntegralMonitor(
        surfaces=["surf1", "surf2"], output_fields=["Cp", "qcriterion"]
    )

    assert integral

    output = MonitorOutput(
        output_fields=["Cp", "qcriterion"], monitors={"m1": probe, "m2": integral}
    )

    assert output

    to_file_from_file_test(output)
