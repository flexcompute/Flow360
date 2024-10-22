import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air, Sutherland
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    HeatFlux,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)


def create_heat_flux_cylinder_freestream_surfaces():
    return [
        Surface(name="fluid/inlet"),
        Surface(name="fluid/outlet"),
        Surface(name="fluid/top"),
        Surface(name="fluid/bottom"),
    ]


def create_heat_flux_cylinder_slip_wall_surfaces():
    return [Surface(name="fluid/periodic_0_l"), Surface(name="fluid/periodic_0_r")]


def append_heat_flux_cylinder_boundaries(params):
    params.models.append(Freestream(entities=create_heat_flux_cylinder_freestream_surfaces()))
    params.models.append(SlipWall(entities=create_heat_flux_cylinder_slip_wall_surfaces()))
    my_wall = Surface(name="fluid/wall")
    with SI_unit_system:
        default_thermal_state = ThermalState()
    params.models.append(
        Wall(
            surfaces=my_wall,
            heat_spec=HeatFlux(
                -0.001 * default_thermal_state.density * default_thermal_state.speed_of_sound**3
            ),
        )
    )


def create_heat_flux_cylinder_base_param():

    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 1 * u.m
        viscosity = viscosity_from_muRef(
            0.005, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=(0, 0, 0),
                moment_length=(1, 1, 1) * u.m,
                area=20.0 * u.m * u.m,
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=0.1,
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
                        kappa_MUSCL=-1,
                        linear_solver=LinearSolver(max_iterations=25),
                    ),
                    turbulence_model_solver=NoneSolver(),
                ),
            ],
            time_stepping=Steady(CFL=RampCFL(initial=100, final=100, ramp_steps=5), max_steps=1000),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=[
                        "primitiveVars",
                        "residualNavierStokes",
                        "T",
                    ],
                ),
                SurfaceOutput(
                    entities=[Surface(name="fluid/wall")],
                    output_format="paraview",
                    output_fields=["Cp", "primitiveVars", "T", "heatFlux"],
                ),
            ],
        )
    return param


@pytest.fixture
def create_heat_flux_cylinder_param():
    param = create_heat_flux_cylinder_base_param()
    append_heat_flux_cylinder_boundaries(param)
    return param
