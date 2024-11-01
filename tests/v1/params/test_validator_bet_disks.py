import copy
import unittest

import pytest

import flow360.component.v1xxx as fl
from flow360.component.v1.flow360_params import (
    BETDisk,
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
    Flow360Params,
)

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


twists = [BETDiskTwist(radius=0.3, twist=0.04), BETDiskTwist(radius=0.6, twist=0.08)]
chords = [BETDiskChord(radius=0.3, chord=0.1), BETDiskChord(radius=0.6, chord=0.2)]
lift_coeffs = [[[0.1, 0.2]]]
drag_coeffs = [[[0.01, 0.02]]]
polars = [BETDiskSectionalPolar(lift_coeffs=lift_coeffs, drag_coeffs=drag_coeffs)]


def test_bet_disks_good():
    with fl.SI_unit_system:
        param = Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            bet_disks=[
                BETDisk(
                    alphas=[-2, 5],
                    center_of_rotation=(1, 2, 3),
                    axis_of_rotation=(1, 2, 3),
                    number_of_blades=3,
                    radius=1,
                    omega=0.2,
                    chord_ref=0.1,
                    thickness=0.04,
                    n_loading_nodes=20,
                    mach_numbers=[0.4],
                    reynolds_numbers=[10000],
                    twists=[BETDiskTwist()] * 3,
                    chords=[BETDiskChord()] * 3,
                    sectional_polars=[BETDiskSectionalPolar()] * 3,
                    sectional_radiuses=[0.1, 0.3, 0.8],
                ),
            ],
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
        )
        param.flow360_json()

        param = Flow360Params(
            bet_disks=[
                BETDisk(
                    alphas=[-2, 5],
                    center_of_rotation=(1, 2, 3),
                    axis_of_rotation=(1, 2, 3),
                    number_of_blades=3,
                    radius=1,
                    omega=0.2,
                    chord_ref=0.1,
                    thickness=0.04,
                    n_loading_nodes=20,
                    mach_numbers=[0.4],
                    reynolds_numbers=[10000],
                    twists=twists,
                    chords=chords,
                    sectional_polars=polars,
                    sectional_radiuses=[0.1],
                ),
            ],
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )


def test_bet_disks_alphas_disorder():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match="alphas are not in increasing order.",
        ):
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[5, -2],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )


def test_bet_disks_duplicated_radial_locations():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match="BET disk has duplicated radius at .* in chords.",
        ):
            chords_wrong = copy.deepcopy(chords)
            chords_wrong[0].radius = chords_wrong[1].radius
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[-2, 5],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords_wrong,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        with pytest.raises(
            ValueError,
            match="BET disk has duplicated radius at .* in twists.",
        ):
            twists_wrong = copy.deepcopy(twists)
            twists_wrong[0].radius = twists_wrong[1].radius
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[-2, 5],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists_wrong,
                        chords=chords,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )


def test_bet_disks_number_of_polars():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match=r"length of sectional_radiuses \(1\) is not the same as that of sectional_polars \(2\)",
        ):
            polars_wrong = copy.deepcopy(polars)
            polars_wrong.append(polars[0])
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[-2, 5],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars_wrong,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )


def test_bet_disks_dimension_polars():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match=r"\(cross section: 0\): number of MachNumbers = 2, but the first dimension of lift_coeffs is 1.",
        ):
            mach_wrong = [0.4, 0.5]
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[-2, 5],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=mach_wrong,
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match=r"\(cross section: 0\) \(Mach index \(0-based\) 0\): number of Reynolds = 2, but the second dimension of lift_coeffs is 1.",
        ):
            re_wrong = [0.4, 0.5]
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=[-2, 5],
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=re_wrong,
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match=r"\(cross section: 0\) \(Mach index \(0-based\) 0, Reynolds index \(0-based\) 0\): number of Alphas = 3, but the third dimension of lift_coeffs is 2.",
        ):
            alpha_wrong = [0.4, 0.5, 0.6]
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=alpha_wrong,
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match=r"\(cross section: 0\) \(Mach index \(0-based\) 0, Reynolds index \(0-based\) 0\): number of Alphas = 3, but the third dimension of drag_coeffs is 2.",
        ):
            lift_coeffs_local = [[[0.1, 0.2, 0.3]]]
            drag_coeffs_local = [[[0.01, 0.02]]]
            polars_local = [
                BETDiskSectionalPolar(lift_coeffs=lift_coeffs_local, drag_coeffs=drag_coeffs_local)
            ]
            alpha_wrong = [0.4, 0.5, 0.6]
            Flow360Params(
                bet_disks=[
                    BETDisk(
                        alphas=alpha_wrong,
                        center_of_rotation=(1, 2, 3),
                        axis_of_rotation=(1, 2, 3),
                        number_of_blades=3,
                        radius=1,
                        omega=0.2,
                        chord_ref=0.1,
                        thickness=0.04,
                        n_loading_nodes=20,
                        mach_numbers=[0.4],
                        reynolds_numbers=[10000],
                        twists=twists,
                        chords=chords,
                        sectional_polars=polars_local,
                        sectional_radiuses=[0.1],
                    ),
                ],
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
