import pytest
from numpy import pi

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    AngularVelocity,
    Rotation,
)
from flow360.component.simulation.primitives import Cylinder
from tests.simulation.translator.utils.xv15_bet_disk_helper import (
    createBETDiskUnsteady,
    createUnsteadyTimeStepping,
)
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    _BET_cylinder,
    create_param_base,
)


@pytest.fixture
def cylinder_inner():
    return Cylinder(
        name="inner-but-not-full-name",
        private_attribute_full_name="inner",
        center=(0, 0, 0) * u.inch,
        axis=[0, 0, 1],
        # filler values
        outer_radius=50 * u.inch,
        height=15 * u.inch,
    )


@pytest.fixture
def cylinder_middle():
    return Cylinder(
        name="middle-but-not-full-name",
        private_attribute_full_name="middle",
        center=(0, 0, 0) * u.inch,
        axis=[0, 0, 1],
        inner_radius=50 * u.inch,
        outer_radius=100 * u.inch,
        height=15 * u.inch,
    )


@pytest.fixture
def create_nested_rotation_param(cylinder_inner, cylinder_middle):
    """
    params = runCase_unsteady_hover(
        pitch_in_degree=10,
        rpm_bet=294.25225,
        rpm_inner=-127.93576086956521,
        rpm_middle=-166.31648913043477,
        CTRef=0.008073477299631027,
        CQRef=0.0007044185338787385,
        tolerance=1.25e-2,
        caseName="runCase_unsteady_hover-2"
    )
    """
    rpm_bet = 294.25225
    rpm_inner = -127.93576086956521
    rpm_middle = -166.31648913043477
    mesh_unit = 1 * u.inch
    params = create_param_base()
    bet_disk = createBETDiskUnsteady(_BET_cylinder, 10, rpm_bet)
    rotation_inner = Rotation(
        volumes=[cylinder_inner],
        spec=AngularVelocity(rpm_inner * u.rpm),
        parent_volume=cylinder_middle,
        isMRF=False,
    )
    omega_middle = (
        rpm_middle
        / 60
        * 2
        * pi
        * u.radian
        / u.s
        * mesh_unit
        / params.operating_condition.thermal_state.speed_of_sound
    )
    rotation_middle = Rotation(
        entities=[cylinder_middle],
        spec=AngleExpression(str(omega_middle.v.item()) + "*t"),
        isMRF=False,
    )
    params.models += [bet_disk, rotation_inner, rotation_middle]
    params.time_stepping = createUnsteadyTimeStepping(rpm_bet - rpm_inner - rpm_middle)
    params.time_stepping.max_pseudo_steps = 30
    return params
