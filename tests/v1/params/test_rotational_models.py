import unittest

import numpy as np
import pytest

import flow360.component.v1.modules as fl
from flow360.component.v1.flow360_params import (
    ActuatorDisk,
    BETDisk,
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
    ForcePerArea,
)
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

    with pytest.raises(ValueError):
        fpa = ForcePerArea(radius=[0, 1, 3], thrust=[1, 1], circumferential=[1, 1])

    to_file_from_file_test(ad)
    compare_to_ref(ad, "../ref/case_params/actuator_disk/json.json")
    compare_to_ref(ad, "../ref/case_params/actuator_disk/yaml.yaml")

    ad.axis_thrust = (1, 2, 5)
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            actuator_disks=[ad],
            fluid_properties=fl.air,
            geometry=fl.Geometry(meshUnit=1),
            boundaries={
                "MyBC": fl.FreestreamBoundary(),
            },
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(),
        )
        solver_params = params.to_solver()
        assert (
            abs(np.linalg.norm(np.array(solver_params.actuator_disks[0].axis_thrust)) - 1) < 1e-10
        )


def test_bet_disk():
    twist1 = BETDiskTwist(radius=0, twist=0)
    twist2 = BETDiskTwist(radius=0.5, twist=0.25)
    twist3 = BETDiskTwist(radius=1, twist=0.4)

    assert twist1 and twist2 and twist3

    chord1 = BETDiskChord(radius=0, chord=0.5)
    chord2 = BETDiskChord(radius=0.5, chord=0.7)
    chord3 = BETDiskChord(radius=1, chord=0.65)

    assert chord1 and chord2 and chord3

    cl = np.random.rand(4, 4, 6).tolist()
    cd = (np.random.rand(4, 4, 6) * 0.1).tolist()

    for i in range(0, 4):
        for j in range(0, 4):
            cl[i][j][0] = 1.23
            cd[i][j][0] = 0.123
            cl[i][j][-1] = 1.23 * 1.1
            cd[i][j][-1] = 0.123 * 1.1

    polar1 = BETDiskSectionalPolar(lift_coeffs=cl, drag_coeffs=cd)
    polar2 = BETDiskSectionalPolar(lift_coeffs=cl, drag_coeffs=cd)
    polar3 = BETDiskSectionalPolar(lift_coeffs=cl, drag_coeffs=cd)

    assert polar1 and polar2 and polar3

    with fl.SI_unit_system:
        bet = BETDisk(
            rotation_direction_rule="leftHand",
            center_of_rotation=(0, 0, 0),
            axis_of_rotation=(2, 0, 0),
            number_of_blades=4,
            radius=0.5,
            omega=0.75,
            chord_ref=0.5,
            thickness=0.5,
            n_loading_nodes=6,
            mach_numbers=[0.1, 0.2, 0.3, 0.4],
            reynolds_numbers=[1e4, 1e5, 1e6, 1e7],
            alphas=[-180 + 1e-10, 15, 20, 25, 30, 180 - 1e-10],
            twists=[twist1, twist2, twist3],
            chords=[chord1, chord2, chord3],
            sectional_polars=[polar1, polar2, polar3],
            sectional_radiuses=[0, 0.5, 1],
        )
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(meshUnit=1),
            boundaries={
                "MyBC": fl.FreestreamBoundary(),
            },
            bet_disks=[bet],
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(),
        )
        solver_params = params.to_solver()
        assert bet
        bet_disk = solver_params.bet_disks[0]
        for polarItem in bet_disk.sectional_polars:
            for coeff2D in polarItem.lift_coeffs:
                for coeff1D in coeff2D:
                    assert coeff1D[0] == coeff1D[-1]
            for coeff2D in polarItem.drag_coeffs:
                for coeff1D in coeff2D:
                    assert coeff1D[0] == coeff1D[-1]
        assert abs(np.linalg.norm(np.array(bet_disk.axis_of_rotation)) - 1) < 1e-10

    to_file_from_file_test(bet)
