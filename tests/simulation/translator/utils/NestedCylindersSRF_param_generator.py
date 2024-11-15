import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air, Sutherland
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import (
    AngularVelocity,
    Fluid,
    Rotation,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import AdaptiveCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)


@pytest.fixture
def srf_cylinder():
    return Cylinder(
        name="blk-1",
        center=(0, 0, -1.32392) * u.m,
        axis=[0, 0, -1],
        # filler values
        outer_radius=2 * u.m,
        height=10.0 * u.m,
    )


@pytest.fixture
def create_NestedCylindersSRF_param(srf_cylinder):
    inner_wall = Surface(name="blk-1/Cylinder")
    outer_wall = Surface(name="blk-1/OuterWall")
    my_freestream = Surface(name="blk-1/InletOutlet")
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 1 * u.m
        viscosity = viscosity_from_muRef(
            0.005, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.05,
                alpha=-90 * u.deg,
                thermal_state=ThermalState(
                    material=Air(
                        dynamic_viscosity=Sutherland(
                            reference_temperature=default_thermal_state.temperature,
                            reference_viscosity=viscosity,
                            effective_temperature=default_thermal_state.material.dynamic_viscosity.effective_temperature,
                        )
                    ),
                ),
            ),
            models=[
                Fluid(
                    navier_stokes_solver=NavierStokesSolver(
                        absolute_tolerance=1e-9,
                        linear_solver=LinearSolver(max_iterations=35),
                    ),
                    turbulence_model_solver=NoneSolver(),
                ),
                Wall(surfaces=[inner_wall], use_wall_function=False),
                Wall(surfaces=[outer_wall], use_wall_function=False, velocity=[0.0, 0.0, 0.0]),
                Freestream(entities=[my_freestream]),
                Rotation(entities=[srf_cylinder], spec=AngularVelocity(812.31 * u.rpm), isMRF=True),
            ],
            time_stepping=Steady(CFL=AdaptiveCFL(), max_steps=2000),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=["primitiveVars", "Mach", "VelocityRelative"],
                ),
                SurfaceOutput(
                    entities=[inner_wall, outer_wall],
                    output_format="paraview",
                    output_fields=["primitiveVars", "Cp", "Cf", "yPlus"],
                ),
            ],
        )
    return param
