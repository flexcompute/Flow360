import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
)
from flow360.component.simulation.models.surface_models import Freestream
from flow360.component.simulation.models.volume_models import BETDisk, Fluid
from flow360.component.simulation.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import Cylinder, ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import imperial_unit_system
from tests.simulation.translator.utils.xv15_bet_disk_helper import (
    createBETDiskSteady,
    createBETDiskUnsteady,
    createSteadyTimeStepping,
    createUDDInstance,
    createUnsteadyTimeStepping,
)


def create_param_base():
    with imperial_unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=(0, 0, 0),
                moment_length=1 * u.inch,
                area=70685.83470577035 * u.inch * u.inch,
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=0,
                alpha=-90 * u.deg,
                thermal_state=ThermalState(),
                reference_mach=0.69,
            ),
            models=[
                Fluid(
                    material=Air(),
                    navier_stokes_solver=NavierStokesSolver(
                        absolute_tolerance=1e-10,
                        linear_solver=LinearSolver(max_iterations=25),
                        kappa_MUSCL=-1.0,
                        update_jacobian_frequency=4,
                    ),
                    turbulence_model_solver=NoneSolver(),
                ),
                Freestream(entities=Surface(name="1")),
            ],
            outputs=[
                VolumeOutput(
                    output_format="tecplot",
                    output_fields=["primitiveVars", "betMetrics", "qcriterion"],
                ),
                SurfaceOutput(
                    output_format="both",
                    output_fields=["primitiveVars", "Cp", "Cf", "CfVec"],
                ),
            ],
        )
    return params


_BET_cylinder = Cylinder(
    name="my_bet_disk_volume",
    center=(0, 0, 0) * u.inch,
    axis=[0, 0, 1],
    outer_radius=150 * u.inch,
    height=15 * u.inch,
)

_rpm_hover_mode = 588.50450
_rpm_airplane_mode = 460.5687


@pytest.fixture
def create_steady_hover_param():
    """
    params = runCase_steady_hover(
        10, rpm_hover_mode, 0.008261, 0.0006287, "runCase_steady_hover-2"
    )
    """
    params = create_param_base()
    bet_disk = createBETDiskSteady(_BET_cylinder, 10, _rpm_hover_mode)
    params.models.append(bet_disk)
    params.time_stepping = createSteadyTimeStepping()
    return params


@pytest.fixture
def create_steady_airplane_param():
    """
    params = runCase_steady_airplane(
        26, rpm_airplane_mode, 0.001987, 0.0007581, "runCase_steady_airplane"
    )
    """
    params = create_param_base()
    bet_disk = createBETDiskSteady(_BET_cylinder, 26, _rpm_airplane_mode)
    params.models.append(bet_disk)
    params.time_stepping = createSteadyTimeStepping()
    return params


@pytest.fixture
def create_unsteady_hover_param():
    """
    params = runCase_unsteady_hover(
        10, rpm_hover_mode, 0.008423, 0.0006905, "runCase_unsteady_hover-2"
    )
    """
    params = create_param_base()
    bet_disk = createBETDiskUnsteady(_BET_cylinder, 10, _rpm_hover_mode)
    params.models.append(bet_disk)
    params.time_stepping = createUnsteadyTimeStepping(_rpm_hover_mode)
    return params


@pytest.fixture
def create_unsteady_hover_UDD_param():
    """
    params = runCase_unsteady_hover_UDD(
        10, rpm_hover_mode, 0.011034, 0.00077698, "runCase_unsteady_hover_UDD"
    )
    """
    params = create_param_base()
    bet_disk = createBETDiskUnsteady(_BET_cylinder, 10, _rpm_hover_mode)
    params.models.append(bet_disk)
    params.user_defined_dynamics = [createUDDInstance()]
    params.time_stepping = createUnsteadyTimeStepping(_rpm_hover_mode)
    return params
