import numpy as np

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import (
    BETDisk,
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
)
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.time_stepping.time_stepping import (
    RampCFL,
    Steady,
    Unsteady,
)
from flow360.component.simulation.unit_system import imperial_unit_system
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)

polar_radial_locations = [13.5, 25.5, 37.5, 76.5, 120, 150]

radial_loc_for_twist = [13.5, 25.5, 37.356855, 76.5, 120, 150]
radial_twists_pitch_0 = [
    30.29936539609504,
    26.047382700278234,
    21.01189770991256,
    6.596477306554306,
    -1.5114259546742872,
    -6.02484,
]
radial_loc_for_chord = [13.4999999, 13.5, 37.356855462705056, 150.0348189415042]
radial_chords = [0, 17.69622361, 14.012241185039136, 14.004512929656503]

mach_numbers = [0]
reynolds_numbers = [1000000]

sectionalPolars = [
    {
        "liftCoeffs": [
            [
                [
                    -0.4805,
                    -0.3638,
                    -0.2632,
                    -0.162,
                    -0.0728,
                    -0.0045,
                    0.0436,
                    0.2806,
                    0.4874,
                    0.6249,
                    0.7785,
                    0.9335,
                    1.0538,
                    1.1929,
                    1.302,
                    1.4001,
                    1.4277,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.04476,
                    0.0372,
                    0.02956,
                    0.02272,
                    0.01672,
                    0.01157,
                    0.00757,
                    0.00762,
                    0.00789,
                    0.00964,
                    0.01204,
                    0.01482,
                    0.01939,
                    0.02371,
                    0.02997,
                    0.03745,
                    0.05013,
                ]
            ]
        ],
    },
    {
        "liftCoeffs": [
            [
                [
                    -0.875,
                    -0.7769,
                    -0.6771,
                    -0.5777,
                    -0.4689,
                    -0.3714,
                    -0.1894,
                    0.0577,
                    0.3056,
                    0.552,
                    0.7908,
                    1.0251,
                    1.2254,
                    1.3668,
                    1.4083,
                    1.3681,
                    1.307,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.03864,
                    0.03001,
                    0.0226,
                    0.01652,
                    0.01212,
                    0.00933,
                    0.00623,
                    0.00609,
                    0.00622,
                    0.00632,
                    0.00666,
                    0.00693,
                    0.00747,
                    0.00907,
                    0.01526,
                    0.02843,
                    0.04766,
                ]
            ]
        ],
    },
    {
        "liftCoeffs": [
            [
                [
                    -0.875,
                    -0.7769,
                    -0.6771,
                    -0.5777,
                    -0.4689,
                    -0.3714,
                    -0.1894,
                    0.0577,
                    0.3056,
                    0.552,
                    0.7908,
                    1.0251,
                    1.2254,
                    1.3668,
                    1.4083,
                    1.3681,
                    1.307,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.03864,
                    0.03001,
                    0.0226,
                    0.01652,
                    0.01212,
                    0.00933,
                    0.00623,
                    0.00609,
                    0.00622,
                    0.00632,
                    0.00666,
                    0.00693,
                    0.00747,
                    0.00907,
                    0.01526,
                    0.02843,
                    0.04766,
                ]
            ]
        ],
    },
    {
        "liftCoeffs": [
            [
                [
                    -1.3633,
                    -1.2672,
                    -1.1603,
                    -1.0317,
                    -0.8378,
                    -0.6231,
                    -0.4056,
                    -0.1813,
                    0.0589,
                    0.2997,
                    0.5369,
                    0.7455,
                    0.9432,
                    1.1111,
                    1.2157,
                    1.3208,
                    1.4081,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.02636,
                    0.01909,
                    0.01426,
                    0.01174,
                    0.00999,
                    0.00864,
                    0.00679,
                    0.00484,
                    0.00469,
                    0.00479,
                    0.00521,
                    0.00797,
                    0.01019,
                    0.01213,
                    0.0158,
                    0.02173,
                    0.03066,
                ]
            ]
        ],
    },
    {
        "liftCoeffs": [
            [
                [
                    -1.4773,
                    -1.3857,
                    -1.2144,
                    -1.0252,
                    -0.8141,
                    -0.5945,
                    -0.3689,
                    -0.1411,
                    0.0843,
                    0.3169,
                    0.5378,
                    0.7576,
                    0.973,
                    1.1754,
                    1.3623,
                    1.4883,
                    1.5701,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.02328,
                    0.01785,
                    0.01414,
                    0.01116,
                    0.00925,
                    0.00782,
                    0.00685,
                    0.00585,
                    0.00402,
                    0.00417,
                    0.00633,
                    0.00791,
                    0.00934,
                    0.01102,
                    0.01371,
                    0.01744,
                    0.02419,
                ]
            ]
        ],
    },
    {
        "liftCoeffs": [
            [
                [
                    -1.4773,
                    -1.3857,
                    -1.2144,
                    -1.0252,
                    -0.8141,
                    -0.5158,
                    -0.3378,
                    -0.1137,
                    0.11,
                    0.3331,
                    0.5499,
                    0.77,
                    0.9846,
                    1.1893,
                    1.3707,
                    1.4427,
                    1.4346,
                ]
            ]
        ],
        "dragCoeffs": [
            [
                [
                    0.02328,
                    0.01785,
                    0.01414,
                    0.01116,
                    0.00925,
                    0.03246,
                    0.00696,
                    0.00565,
                    0.00408,
                    0.00399,
                    0.0061,
                    0.00737,
                    0.00923,
                    0.01188,
                    0.01665,
                    0.02778,
                    0.04379,
                ]
            ]
        ],
    },
]


def _createBETTwistsAndChords(pitch_in_degree):
    radial_twists_curr = [twist + pitch_in_degree for twist in radial_twists_pitch_0]
    twists = []
    chords = []
    with imperial_unit_system:
        for radius, twist in zip(radial_loc_for_twist, radial_twists_curr):
            betDiskTwist = BETDiskTwist(radius=radius, twist=twist)
            twists.append(betDiskTwist)
        for radius, chord in zip(radial_loc_for_chord, radial_chords):
            betDiskChord = BETDiskChord(radius=radius, chord=chord)
            chords.append(betDiskChord)
    return twists, chords


def _createBETPolars():
    sectional_radiuses = []
    polars = []
    with imperial_unit_system:
        for radial_index, radial_loc in enumerate(polar_radial_locations):
            sectional_radiuses.append(radial_loc)
            cl3d = sectionalPolars[radial_index]["liftCoeffs"]
            cd3d = sectionalPolars[radial_index]["dragCoeffs"]
            polar_curr_section = BETDiskSectionalPolar(lift_coeffs=cl3d, drag_coeffs=cd3d)
            polars.append(polar_curr_section)
    return sectional_radiuses, polars


def createBETDiskSteady(cylinder_entity: Cylinder, pitch_in_degree, rpm):
    alphas = np.arange(-16, 18, 2, dtype=int)
    sectional_radiuses, sectional_polars = _createBETPolars()
    twists, chords = _createBETTwistsAndChords(pitch_in_degree)
    with imperial_unit_system:
        betDisk = BETDisk(
            entities=[cylinder_entity],
            rotation_direction_rule="leftHand",
            number_of_blades=3,
            omega=rpm * u.rpm,
            chord_ref=14 * u.inch,
            n_loading_nodes=20,
            mach_numbers=mach_numbers,
            reynolds_numbers=reynolds_numbers,
            twists=twists,
            chords=chords,
            alphas=alphas.tolist(),
            sectional_radiuses=sectional_radiuses,
            sectional_polars=sectional_polars,
        )
    return betDisk


def createBETDiskUnsteady(cylinder_entity: Cylinder, pitch_in_degree, rpm):
    bet_disk = createBETDiskSteady(cylinder_entity, pitch_in_degree, rpm)
    bet_disk.blade_line_chord = 25 * u.inch
    bet_disk.initial_blade_direction = (1, 0, 0)
    return bet_disk


def createSteadyTimeStepping():
    with imperial_unit_system:
        time_stepping = Steady(max_steps=10000, CFL=RampCFL(initial=1, final=100, ramp_steps=500))
    return time_stepping


def createUnsteadyTimeStepping(rpm):
    def dt_to_revolve_one_degree(rpm):
        return (1.0 / (rpm / 60 * 360)) * u.s

    with imperial_unit_system:
        time_stepping = Unsteady(
            steps=1800,
            step_size=2 * dt_to_revolve_one_degree(rpm),
            max_pseudo_steps=25,
            CFL=RampCFL(initial=100, final=10000, ramp_steps=15),
        )
    return time_stepping


def createUDDInstance():
    udd = UserDefinedDynamic(
        name="BET_Controller",
        input_vars=["bet_0_thrust"],
        output_vars={"bet_0_omega": "state[0];"},
        constants={
            "ThrustTarget": 300,
            "PConst": 1e-7,
            "IConst": 1e-7,
            "omega0": 0.003,
        },
        state_vars_initial_value=["0.003", "0.0", "0", "0", "0"],
        update_law=[
            "if (physicalStep > 150 and pseudoStep == 0) PConst * (ThrustTarget - bet_0_thrust)  + IConst * state[1] + omega0; else state[0];",
            "if (physicalStep > 150 and pseudoStep == 0) state[1] + (ThrustTarget - bet_0_thrust); else state[1];",
            "(physicalStep > 150 and pseudoStep == 0)",
            "ThrustTarget - bet_0_thrust",
            "IConst * state[1]",
        ],
    )
    return udd
