import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.params import MeshingParameters
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.models.surface_models import HeatFlux, SlipWall, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition import (
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import Box, ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import CGS_unit_system
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from tests.utils import to_file_from_file_test


@pytest.mark.usefixtures("array_equality_override")
def test_simulation_params_seralization():
    my_wall_surface = Surface(name="my_wall")
    my_slip_wall_surface = Surface(name="my_slip_wall")
    with CGS_unit_system:
        my_box = Box(
            name="my_box",
            center=(1.2, 2.3, 3.4) * u.m,
            size=(1.0, 2.0, 3.0) * u.m,
            axes=((1, 0, 1), (1, 2, 3)),
        )
        param = SimulationParams(
            meshing=MeshingParameters(
                farfield="auto",
                refinement_factor=1.0,
                gap_treatment_strength=0.5,
                surface_layer_growth_rate=1.5,
                refinements=[UniformRefinement(entities=[my_box], spacing=0.1 * u.m)],
            ),
            reference_geometry=ReferenceGeometry(
                moment_center=(1, 2, 3)  # , moment_length=1.0 * u.m, area=1.0 * u.cm**2
            ),
            operating_condition=GenericReferenceCondition(
                velocity_magnitude=234,
                thermal_state=ThermalState(temperature=300 * u.K, density=1 * u.g / u.cm**3),
            ),
            models=[
                Fluid(),
                Wall(
                    entities=[my_wall_surface],
                    use_wall_function=True,
                    velocity=(1.0, 1.2, 2.4) * u.ft / u.s,
                    heat_spec=HeatFlux(1.0 * u.W / u.m**2),
                ),
                SlipWall(entities=[my_slip_wall_surface]),
            ],
            time_stepping=Steady(),
            user_defined_dynamics=[
                UserDefinedDynamic(
                    name="fake",
                    input_vars=["fake"],
                    constants={"ff": 123},
                    state_vars_initial_value=["fake"],
                    update_law=["fake"],
                )
            ],
        )
    to_file_from_file_test(param)
