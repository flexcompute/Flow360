import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air, Sutherland
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    LinearSolver,
    NavierStokesSolver,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid, Rotation
from flow360.component.simulation.operating_condition import (
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
def rotation_cylinder():
    return Cylinder(
        name="innerRotating",
        center=(0, 0, 0) * u.m,
        axis=[0, 0, -1],
        # filler values
        outer_radius=4.1088 * u.m,
        height=1.0 * u.m,
    )


@pytest.fixture
def create_XV15HoverMRF_param(rotation_cylinder):
    my_wall = Surface(name="innerRotating/blade")
    my_freestream = Surface(name="farField/farField")
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 1 * u.m
        viscosity = viscosity_from_muRef(
            4.29279e-08, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=1.46972e-02,
                reference_mach=0.70,
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
                    turbulence_model_solver=KOmegaSST(
                        absolute_tolerance=1e-8,
                        linear_solver=LinearSolver(max_iterations=25),
                    ),
                ),
                Wall(surfaces=[my_wall], use_wall_function=False),
                Freestream(entities=[my_freestream]),
                Rotation(entities=[rotation_cylinder], spec=600.106 * u.rpm),
            ],
            time_stepping=Steady(CFL=AdaptiveCFL(), max_steps=4000),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=["primitiveVars", "Mach", "qcriterion", "nuHat"],
                ),
                SurfaceOutput(
                    entities=[my_wall],
                    output_format="paraview",
                    output_fields=["primitiveVars", "Cp", "Cf", "yPlus"],
                ),
            ],
        )
    return param
