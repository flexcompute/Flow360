import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import Freestream
from flow360.component.simulation.models.volume_models import ActuatorDisk, ForcePerArea
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import Cylinder, ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture
def actuator_disk_create_param():
    with SI_unit_system:
        ts = ThermalState()
        acoustics_pressure = ts.density * ts.speed_of_sound**2
        fpa = ForcePerArea(
            radius=[0.01, 0.05, 0.1],
            thrust=[0.001, 0.02, 0] * acoustics_pressure,
            circumferential=[-0.0001, -0.003, 0] * acoustics_pressure,
        )
        assert fpa

        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(0, 0, 1.0),
            center=(0.0, 0.0, 0.0),
            height=0.01,
            outer_radius=5.0,
        )

        params = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0,
                alpha=-90 * u.deg,
                reference_mach=0.69,
            ),
            models=[
                ActuatorDisk(volumes=[my_cylinder_1], force_per_area=fpa),
                Freestream(entities=Surface(name="1")),
            ],
        )

    return params


@pytest.fixture
def actuator_disk_with_reference_velocity_param():
    with SI_unit_system:
        ts = ThermalState()
        acoustics_pressure = ts.density * ts.speed_of_sound**2
        fpa = ForcePerArea(
            radius=[0.01, 0.05, 0.1],
            thrust=[0.001, 0.02, 0] * acoustics_pressure,
            circumferential=[-0.0001, -0.003, 0] * acoustics_pressure,
        )

        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(0, 0, 1.0),
            center=(0.0, 0.0, 0.0),
            height=0.01,
            outer_radius=5.0,
        )

        params = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0,
                alpha=-90 * u.deg,
                reference_mach=0.69,
            ),
            models=[
                ActuatorDisk(
                    volumes=[my_cylinder_1],
                    force_per_area=fpa,
                    reference_velocity=(100.0, 50.0, 0.0) * u.m / u.s,
                ),
                Freestream(entities=Surface(name="1")),
            ],
        )

    return params
