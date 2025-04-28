import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air, Sutherland
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import AdaptiveCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)


@pytest.fixture
def create_symmetryBC_param():
    my_wall = Surface(name="fluid/wall")
    my_symmetry_plane = Surface(name="fluid/symmetry")
    my_freestream = Surface(name="fluid/farfield")
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 1 * u.m
        viscosity = viscosity_from_muRef(
            4.2925193198151646e-8, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.2,
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
                        absolute_tolerance=1e-10,
                        linear_solver=LinearSolver(max_iterations=35),
                    ),
                    turbulence_model_solver=SpalartAllmaras(
                        absolute_tolerance=1e-9,
                        linear_solver=LinearSolver(max_iterations=25),
                    ),
                ),
                Wall(surfaces=[my_wall], use_wall_function=False),
                SymmetryPlane(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
            ],
            time_stepping=Steady(
                CFL=AdaptiveCFL(max=10000, max_relative_change=1, convergence_limiting_factor=0.25),
                max_steps=5000,
            ),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=[
                        "primitiveVars",
                        "Mach",
                        "gradW",
                    ],
                ),
                SurfaceOutput(
                    entities=[my_wall],
                    output_format="paraview",
                    output_fields=["primitiveVars", "Cp", "Cf", "yPlus"],
                ),
            ],
        )
    return param
