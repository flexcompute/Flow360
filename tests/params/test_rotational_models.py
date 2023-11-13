import unittest

import pytest

from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    BETDisk,
    ForcePerArea,
)
from flow360.component.flow360_params.flow360_temp import (
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
    RotationDirectionRule,
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


def test_bet_disk():
    twist1 = BETDiskTwist(radius=0, twist=0)
    twist2 = BETDiskTwist(radius=0.5, twist=0.25)
    twist3 = BETDiskTwist(radius=1, twist=0.4)

    assert twist1 and twist2 and twist3

    chord1 = BETDiskChord(radius=0, chord=0.5)
    chord2 = BETDiskChord(radius=0.5, chord=0.7)
    chord3 = BETDiskChord(radius=1, chord=0.65)

    assert chord1 and chord2 and chord3

    polar1 = BETDiskSectionalPolar(lift_coeffs=[[[0.5]]], drag_coeffs=[[[0.5]]])
    polar2 = BETDiskSectionalPolar(lift_coeffs=[[[0.5]]], drag_coeffs=[[[0.5]]])
    polar3 = BETDiskSectionalPolar(lift_coeffs=[[[0.5]]], drag_coeffs=[[[0.5]]])

    assert polar1 and polar2 and polar3

    bet = BETDisk(
        rotation_direction_rule=RotationDirectionRule.LEFT_HAND,
        center_of_rotation=(0, 0, 0),
        axis_of_rotation=(0, 1, 0),
        number_of_blades=4,
        radius=0.5,
        omega=0.75,
        chord_ref=0.5,
        thickness=0.5,
        n_loading_nodes=6,
        mach_numbers=[0.1, 0.2, 0.3, 0.4],
        reynolds_numbers=[1e4, 1e5, 1e6, 1e7],
        alphas=[15, 20, 25, 30],
        twists=[twist1, twist2, twist3],
        chords=[chord1, chord2, chord3],
        sectional_polars=[polar1, polar2, polar3],
        sectional_radiuses=[0, 0.5, 1],
    )

    assert bet

    to_file_from_file_test(bet)
