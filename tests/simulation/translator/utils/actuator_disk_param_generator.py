import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import Freestream
from flow360.component.simulation.models.volume_models import ActuatorDisk, ForcePerArea
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.primitives import Cylinder, ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import imperial_unit_system


@pytest.fixture
def actuator_disk_create_param():

    with imperial_unit_system:
        fpa = ForcePerArea(radius=[0, 1, 2, 4], thrust=[1, 1, 2, 2], circumferential=[1, 1, 3, 4])
        assert fpa

        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(5, 0, 0),
            center=(1.2, 2.3, 3.4),
            height=3.0,
            outer_radius=5.0,
        )

        params = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0,
                alpha=-90 * u.deg,
                reference_mach=0.69,
            ),
            reference_geometry=ReferenceGeometry(),
            models=[
                ActuatorDisk(volumes=[my_cylinder_1], force_per_area=fpa),
                Freestream(entities=Surface(name="1")),
            ],
        )

    return params
