import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Wall,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def get_2dcrm_tutorial_param():
    with SI_unit_system:
        my_wall = Surface(name="1")
        my_symmetry_plane = Surface(name="2")
        my_freestream = Surface(name="3")
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=[0.25, 0.005, 0], moment_length=[1, 1, 1], area=0.01
            ),
            operating_condition=AerospaceCondition.from_mach_reynolds(
                mach=0.2,
                reynolds=5e6,
                temperature=272.1 * u.K,
                alpha=16 * u.deg,
                beta=0 * u.deg,
                characteristic_length=1 * u.m,
            ),
            models=[
                Wall(surfaces=[my_wall]),
                SlipWall(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
            ],
        )

    return param


@pytest.fixture()
def get_2dcrm_tutorial_param_deg_c():
    with SI_unit_system:
        my_wall = Surface(name="1")
        my_symmetry_plane = Surface(name="2")
        my_freestream = Surface(name="3")
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=[0.25, 0.005, 0], moment_length=[1, 1, 1], area=0.01
            ),
            operating_condition=AerospaceCondition.from_mach_reynolds(
                mach=0.2,
                reynolds=5e6,
                temperature=-1.05 * u.degC,
                alpha=16 * u.deg,
                beta=0 * u.deg,
                characteristic_length=1 * u.m,
            ),
            models=[
                Wall(surfaces=[my_wall]),
                SlipWall(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
            ],
        )

    return param


@pytest.fixture()
def get_2dcrm_tutorial_param_deg_f():
    with SI_unit_system:
        my_wall = Surface(name="1")
        my_symmetry_plane = Surface(name="2")
        my_freestream = Surface(name="3")
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=[0.25, 0.005, 0], moment_length=[1, 1, 1], area=0.01
            ),
            operating_condition=AerospaceCondition.from_mach_reynolds(
                mach=0.2,
                reynolds=5e6,
                temperature=30.11 * u.degF,
                alpha=16 * u.deg,
                beta=0 * u.deg,
                characteristic_length=1 * u.m,
            ),
            models=[
                Wall(surfaces=[my_wall]),
                SlipWall(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
            ],
        )

    return param
