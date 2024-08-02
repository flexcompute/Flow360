import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air, SolidMaterial, Sutherland
from flow360.component.simulation.models.solver_numerics import (
    HeatEquationSolver,
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Temperature,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid, Solid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import GenericVolume, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)


@pytest.fixture
def create_conjugate_heat_transfer_param():
    """
    params = createCase()
    """
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 1 * u.m
        viscosity = viscosity_from_muRef(
            0.00025, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        volumetric_heat_source = (
            0.001
            * default_thermal_state.density
            * default_thermal_state.speed_of_sound**3
            / mesh_unit
        )
        heat_equation_solver = HeatEquationSolver(
            equation_evaluation_frequency=20,
            linear_solver=LinearSolver(absolute_tolerance=1e-15, max_iterations=100),
        )
        solid_zone_1 = GenericVolume(
            name="solid-1",
            private_attribute_zone_boundary_names=["solid-1/adiabatic-1", "solid-1/isothermal-1"],
        )
        solid_zone_2 = GenericVolume(
            name="solid-2",
            private_attribute_zone_boundary_names=["solid-2/adiabatic-2", "solid-2/isothermal-2"],
        )
        solid_zone_3 = GenericVolume(
            name="solid-3", private_attribute_zone_boundary_names=["solid-3/adiabatic-3"]
        )
        solid_zone_4 = GenericVolume(
            name="solid-4", private_attribute_zone_boundary_names=["solid-4/adiabatic-4"]
        )
        copper = SolidMaterial(name="copper", thermal_conductivity=401)
        params = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.01,
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
                        numerical_dissipation_factor=0.01,
                        linear_solver=LinearSolver(max_iterations=50),
                        low_mach_preconditioner=True,
                    ),
                    turbulence_model_solver=NoneSolver(),
                ),
                Solid(
                    entities=[solid_zone_1, solid_zone_2],
                    heat_equation_solver=heat_equation_solver,
                    material=copper,
                ),
                Solid(
                    entities=solid_zone_3,
                    heat_equation_solver=heat_equation_solver,
                    material=copper,
                    volumetric_heat_source=volumetric_heat_source,
                ),
                Solid(
                    entities=solid_zone_4,
                    heat_equation_solver=heat_equation_solver,
                    material=SolidMaterial(
                        name="super_conductive", thermal_conductivity=4010
                    ),  # unrealistic value for testing
                    volumetric_heat_source=volumetric_heat_source,
                ),
                Freestream(entities=Surface(name="fluid/farfield")),
                Wall(
                    entities=[
                        Surface(
                            name="isothermal-1", private_attribute_full_name="solid-1/isothermal-1"
                        ),
                        Surface(
                            name="isothermal-2", private_attribute_full_name="solid-2/isothermal-2"
                        ),
                    ],
                    heat_spec=Temperature(350 * u.K),
                ),
                Wall(
                    entities=[
                        Surface(
                            name="adiabatic-1", private_attribute_full_name="solid-1/adiabatic-1"
                        ),
                        Surface(
                            name="adiabatic-2", private_attribute_full_name="solid-2/adiabatic-2"
                        ),
                        Surface(
                            name="adiabatic-3", private_attribute_full_name="solid-3/adiabatic-3"
                        ),
                        Surface(
                            name="adiabatic-4", private_attribute_full_name="solid-4/adiabatic-4"
                        ),
                    ]
                ),
                SlipWall(entities=Surface(name="fluid/slipWall")),
            ],
            time_stepping=Steady(max_steps=10000, CFL=RampCFL(initial=1, final=50, ramp_steps=100)),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=["primitiveVars", "residualNavierStokes", "T"],
                ),
                SurfaceOutput(
                    output_format="paraview",
                    output_fields=[
                        "Cp",
                        "primitiveVars",
                        "T",
                        "heatFlux",
                        "lowMachPreconditionerSensor",
                    ],
                ),
            ],
        )
    return params
