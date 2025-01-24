import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Air
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
)
from flow360.component.simulation.models.surface_models import (
    Inflow,
    Outflow,
    Pressure,
    SlipWall,
    TotalPressure,
)
from flow360.component.simulation.models.volume_models import Fluid, PorousMedium
from flow360.component.simulation.operating_condition.operating_condition import (
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import (
    Box,
    GenericVolume,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    viscosity_from_muRef,
)


def _create_porous_media_param(slip_wall_list, inflow, outflow, porous_zone):
    with SI_unit_system:
        mesh_unit = 1 * u.m
        default_thermal_state = ThermalState()
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(),
            operating_condition=GenericReferenceCondition.from_mach(
                mach=0.2,
                thermal_state=ThermalState(
                    material=Air(
                        dynamic_viscosity=viscosity_from_muRef(
                            2e-6, mesh_unit=mesh_unit, thermal_state=default_thermal_state
                        ),
                    )
                ),
            ),
            models=[
                Fluid(
                    navier_stokes_solver=NavierStokesSolver(
                        absolute_tolerance=1e-10,
                        kappa_MUSCL=0.01,
                        linear_solver=LinearSolver(max_iterations=25),
                    ),
                    turbulence_model_solver=NoneSolver(),
                ),
                PorousMedium(
                    entities=[porous_zone],
                    darcy_coefficient=[1e6, 0, 0],
                    forchheimer_coefficient=[1, 0, 0],
                    volumetric_heat_source=0,
                ),
                SlipWall(entities=slip_wall_list),
                Inflow(
                    entities=[inflow],
                    total_temperature=1.008 * default_thermal_state.temperature,
                    spec=TotalPressure(value=1.028281 * default_thermal_state.pressure),
                ),
                Outflow(entities=[outflow], spec=Pressure(default_thermal_state.pressure)),
            ],
            time_stepping=Steady(CFL=RampCFL(initial=1, final=100, ramp_steps=100), max_steps=2000),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=[
                        "primitiveVars",
                        "vorticity",
                        "residualNavierStokes",
                        "T",
                        "s",
                        "Cp",
                        "mut",
                        "mutRatio",
                    ],
                ),
                SurfaceOutput(
                    entities=[slip_wall_list, inflow, outflow],
                    output_format="paraview",
                    output_fields=[
                        "Cp",
                        "Cf",
                        "CfVec",
                        "primitiveVars",
                        "yPlus",
                        "Mach",
                        "wallDistance",
                    ],
                ),
            ],
        )
    return param


@pytest.fixture()
def create_porous_media_box_param():
    slipWall1, slipWall2, slipWall3 = (
        Surface(name="blk-1/zblocks"),
        Surface(name="blk-1/xblocks"),
        Surface(name="blk-1/xblocks2"),
    )
    inflow = Surface(name="blk-1/yblocks")
    outflow = Surface(name="blk-1/yblocks2")
    box = Box.from_principal_axes(
        name="box",
        axes=[[0, 1, 0], [0, 0, 1]],
        center=[0, 0, 0] * u.m,
        size=[0.2, 0.3, 2] * u.m,
    )
    return _create_porous_media_param(
        slip_wall_list=[slipWall1, slipWall2, slipWall3],
        inflow=inflow,
        outflow=outflow,
        porous_zone=box,
    )


@pytest.fixture()
def create_porous_media_volume_zone_param():
    slipWall1, slipWall2, slipWall3 = (
        Surface(name="blk-1/slip"),
        Surface(name="blk-2/slip"),
        Surface(name="blk-3/slip"),
    )
    inflow = Surface(name="blk-3/inflow")
    outflow = Surface(name="blk-1/outflow")
    porous_zone = GenericVolume(
        name="blk-2-but-not-full-name",
        private_attribute_full_name="blk-2",
    )
    porous_zone.axes = [[0, 1, 0], [0, 0, 1]]
    return _create_porous_media_param(
        slip_wall_list=[slipWall1, slipWall2, slipWall3],
        inflow=inflow,
        outflow=outflow,
        porous_zone=porous_zone,
    )
