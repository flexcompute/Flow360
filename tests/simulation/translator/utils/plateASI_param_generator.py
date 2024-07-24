from cmath import pi

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
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    Fluid,
    FromUserDefinedDynamics,
    Rotation,
)
from flow360.component.simulation.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import Cylinder, ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Unsteady
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)

my_farfield = Surface(name="farFieldBlock/farField")
my_slip_walls = [Surface(name="farFieldBlock/slipWall"), Surface(name="plateBlock/slipWall")]
my_wall = Surface(name="plateBlock/noSlipWall")


def append_plateASI_boundaries(param):
    param.models.append(Freestream(entities=[my_farfield]))
    param.models.append(SlipWall(entities=[my_slip_walls]))
    param.models.append(Wall(entities=[my_wall]))


def apply_plateASI_time_stepping(param, deltNonDimensional, physicalSteps):
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 0.1016 * u.m
    delt_SI = deltNonDimensional * (mesh_unit / default_thermal_state.speed_of_sound)
    param.time_stepping = Unsteady(
        max_pseudo_steps=100,
        step_size=delt_SI,
        steps=physicalSteps,
        CFL=RampCFL(final=50000, initial=1, ramp_steps=5),
    )


def apply_plateASI_user_defined_dynamics(param, zeta, K, omegaN, theta0, initOmegaDot, initTheta):
    dynamic = UserDefinedDynamic(
        name="dynamicTheta",
        input_vars=["rotMomentY"],
        constants={
            "I": 0.443768309310345,
            "zeta": zeta,
            "K": K,
            "omegaN": omegaN,
            "theta0": theta0,
        },
        output_vars={
            "omegaDot": "state[0];",
            "omega": "state[1];",
            "theta": "state[2];",
        },
        state_vars_initial_value=[str(initOmegaDot), "0.0", str(initTheta)],
        update_law=[
            "if (pseudoStep == 0) (rotMomentY - K * ( state[2] - theta0 ) - 2 * zeta * omegaN * I *state[1] ) / I; else state[0];",
            "if (pseudoStep == 0) state[1] + state[0] * timeStepSize; else state[1];",
            "if (pseudoStep == 0) state[2] + state[1] * timeStepSize; else state[2];",
        ],
        input_boundary_patches=[my_wall],
        output_target=rotation_cylinder(),
    )
    param.user_defined_dynamics = list()
    param.user_defined_dynamics.append(dynamic)


def rotation_cylinder():
    return Cylinder(
        name="plateBlock",
        center=(0, 0, 0) * u.m,
        axis=[0, 1, 0],
        # filler values
        outer_radius=1.5 * u.m,
        height=0.5625 * u.m,
    )


def add_plateASI_rotation_zone(param):
    param.models.append(Rotation(entities=[rotation_cylinder()], spec=FromUserDefinedDynamics()))


def create_plateASI_base_param(Reynolds, Mach):
    with SI_unit_system:
        default_thermal_state = ThermalState()
        mesh_unit = 0.1016 * u.m
        viscosity = viscosity_from_muRef(
            Mach / Reynolds, mesh_unit=mesh_unit, thermal_state=default_thermal_state
        )
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                moment_center=(0, 0, 0),
                moment_length=(1, 1, 1) * mesh_unit,
                area=0.5325 * mesh_unit * mesh_unit,
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=Mach,
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
                        relative_tolerance=1e-3,
                        linear_solver=LinearSolver(max_iterations=35),
                        kappa_MUSCL=0.33,
                        order_of_accuracy=2,
                        update_jacobian_frequency=1,
                        equation_evaluation_frequency=1,
                    ),
                    turbulence_model_solver=SpalartAllmaras(
                        absolute_tolerance=1e-8,
                        relative_tolerance=1e-2,
                        linear_solver=LinearSolver(max_iterations=35),
                        order_of_accuracy=2,
                        rotation_correction=True,
                        update_jacobian_frequency=1,
                        reconstruction_gradient_limiter=0.5,
                        equation_evaluation_frequency=1,
                    ),
                ),
            ],
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=["primitiveVars"],
                ),
                SurfaceOutput(
                    output_format="paraview",
                    output_fields=["Cp"],
                ),
            ],
        )
    return param


@pytest.fixture
def create_plateASI_param():
    ftToMet = 0.3048
    degreeToRad = pi / 180.0
    LGrid = 0.1016  # m
    Mach = 0.2
    rho_inf = 1.226  # kg/m3
    omega = 51.46413  # rad/s natural frequency
    u_inf = 18
    deltNonDimensional = 1.0
    physicalSteps = 300
    zeta = 4.0
    theta0 = 5 * degreeToRad
    iniPert = 3 * degreeToRad
    Re = u_inf * ftToMet * LGrid / 14.61e-6  # Air @288.15K
    C_inf = u_inf * ftToMet / Mach
    K = 0.0156 / (rho_inf * LGrid**3 * C_inf**2)
    omegaN = omega / (C_inf / LGrid)
    initTheta = theta0 + iniPert  # rad
    initOmegaDot = (0 - K * (initTheta - theta0)) / 0.443768309310345
    Mach = 0.2
    param = create_plateASI_base_param(Re, Mach)
    append_plateASI_boundaries(param)
    apply_plateASI_time_stepping(param, deltNonDimensional, physicalSteps)
    apply_plateASI_user_defined_dynamics(param, zeta, K, omegaN, theta0, initOmegaDot, initTheta)
    add_plateASI_rotation_zone(param)
    return param
