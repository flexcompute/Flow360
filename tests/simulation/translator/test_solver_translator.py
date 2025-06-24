import json
import os
import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.material import Water, aluminum
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    KOmegaSSTModelConstants,
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
    SpalartAllmarasModelConstants,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    Inflow,
    Mach,
    MassFlowRate,
    Outflow,
    Pressure,
    SlaterPorousBleed,
    SlipWall,
    TotalPressure,
    Wall,
    WallRotation,
)
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantities,
)
from flow360.component.simulation.models.volume_models import (
    Fluid,
    NavierStokesInitialCondition,
    NavierStokesModifiedRestartSolution,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    LiquidOperatingCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.output_entities import Slice
from flow360.component.simulation.outputs.outputs import (
    SliceOutput,
    SurfaceOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    GenericVolume,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from tests.simulation.translator.utils.actuator_disk_param_generator import (
    actuator_disk_create_param,
)
from tests.simulation.translator.utils.CHTThreeCylinders_param_generator import (
    create_conjugate_heat_transfer_param,
)
from tests.simulation.translator.utils.heatFluxCylinder_param_generator import (
    create_heat_flux_cylinder_param,
)
from tests.simulation.translator.utils.NestedCylindersSRF_param_generator import (
    create_NestedCylindersSRF_param,
    srf_cylinder,
)
from tests.simulation.translator.utils.om6WingWallModel_params_generator import (
    create_om6wing_wall_model_param,
)
from tests.simulation.translator.utils.plateASI_param_generator import (
    create_plateASI_param,
)
from tests.simulation.translator.utils.porousMedia_param_generator import (
    create_porous_media_box_param,
    create_porous_media_volume_zone_param,
)
from tests.simulation.translator.utils.symmetryBC_param_generator import (
    create_symmetryBC_param,
)
from tests.simulation.translator.utils.TurbFlatPlate137x97_BoxTrip_generator import (
    create_turb_flat_plate_box_trip_param,
)
from tests.simulation.translator.utils.tutorial_2dcrm_param_generator import (
    get_2dcrm_tutorial_param,
    get_2dcrm_tutorial_param_deg_c,
    get_2dcrm_tutorial_param_deg_f,
)
from tests.simulation.translator.utils.vortex_propagation_generator import (
    create_periodic_euler_vortex_param,
    create_vortex_propagation_param,
)
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    create_steady_airplane_param,
    create_steady_hover_param,
    create_unsteady_hover_param,
    create_unsteady_hover_UDD_param,
)
from tests.simulation.translator.utils.xv15BETDiskNestedRotation_param_generator import (
    create_nested_rotation_param,
    cylinder_inner,
    cylinder_middle,
)
from tests.simulation.translator.utils.XV15HoverMRF_param_generator import (
    create_XV15HoverMRF_param,
    rotation_cylinder,
)

assertions = unittest.TestCase("__init__")

from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    HeatEquationInitialCondition,
    Rotation,
    Solid,
)
from flow360.component.simulation.primitives import GenericVolume
from flow360.component.simulation.time_stepping.time_stepping import Unsteady


@pytest.fixture()
def get_om6Wing_tutorial_param():
    my_wall = Surface(name="1")
    my_symmetry_plane = Surface(name="2")
    my_freestream = Surface(name="3")
    with SI_unit_system:
        param = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=0.748844455929999,
                moment_length=0.6460682372650963,
                moment_center=(0, 0, 0),
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
                alpha=3.06 * u.degree,
            ),
            models=[
                Fluid(
                    navier_stokes_solver=NavierStokesSolver(
                        absolute_tolerance=1e-10,
                        linear_solver=LinearSolver(max_iterations=25),
                        kappa_MUSCL=-1.0,
                    ),
                    turbulence_model_solver=SpalartAllmaras(
                        absolute_tolerance=1e-8,
                        linear_solver=LinearSolver(max_iterations=15),
                    ),
                ),
                Wall(surfaces=[my_wall]),
                SlipWall(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
            ],
            time_stepping=Steady(CFL=RampCFL()),
            outputs=[
                VolumeOutput(
                    output_format="paraview",
                    output_fields=[
                        "primitiveVars",
                        "residualNavierStokes",
                        "residualTurbulence",
                        "Mach",
                    ],
                ),
                SliceOutput(
                    slices=[
                        Slice(
                            name="sliceName_1",
                            normal=(0, 1, 0),
                            origin=(0, 0.56413, 0) * u.m,
                        )
                    ],
                    output_format="tecplot",
                    output_fields=[
                        "primitiveVars",
                        "vorticity",
                        "T",
                        "s",
                        "Cp",
                        "mut",
                        "mutRatio",
                        "Mach",
                    ],
                ),
                SurfaceOutput(
                    entities=[my_wall, my_symmetry_plane, my_freestream],
                    output_format="paraview",
                    output_fields=["Cp"],
                ),
            ],
        )
    return param


def translate_and_compare(
    param, mesh_unit, ref_json_file: str, atol=1e-15, rtol=1e-10, debug=False
):
    translated = get_solver_json(param, mesh_unit=mesh_unit)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", ref_json_file)) as fh:
        ref_dict = json.load(fh)
    if debug:
        print(">>> translated = ", translated)
        print("=== translated ===\n", json.dumps(translated, indent=4, sort_keys=True))
        print("=== ref_dict ===\n", json.dumps(ref_dict, indent=4, sort_keys=True))
    assert compare_values(ref_dict, translated, atol=atol, rtol=rtol)


def test_om6wing_tutorial(get_om6Wing_tutorial_param):
    translate_and_compare(
        get_om6Wing_tutorial_param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6Wing.json"
    )


def test_om6wing_temperature(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    params.operating_condition.thermal_state = ThermalState(temperature=15 * u.degC)
    translate_and_compare(
        get_om6Wing_tutorial_param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6Wing.json"
    )

    params.operating_condition.thermal_state = ThermalState(temperature=59 * u.degF)
    translate_and_compare(
        get_om6Wing_tutorial_param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6Wing.json"
    )

    params.operating_condition.thermal_state = ThermalState(temperature=518.67 * u.R)
    translate_and_compare(
        get_om6Wing_tutorial_param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6Wing.json"
    )


def test_om6wing_debug(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    params.models[0].navier_stokes_solver.private_attribute_dict = {"debugType": "minDensity"}
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6Wing_debug_type.json",
    )

    params.models[0].navier_stokes_solver.private_attribute_dict = {"debugPoint": [1.0, 2.0, 3.0]}
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6Wing_debug_point.json",
    )


def test_om6wing_with_specified_freestream_BC(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    params.models[3].turbulence_quantities = TurbulenceQuantities(modified_viscosity_ratio=10)
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_FS_with_turbulence_quantities.json",
    )
    params.models[3].turbulence_quantities = None
    params.models[3].velocity = (01.0, 10.0, 0.0) * u.m / u.s
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_FS_with_vel.json",
    )
    params.models[3].velocity = ["123", "12", "x*y-z"]
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_FS_with_vel_expression.json",
    )


def test_om6wing_with_specified_turbulence_model_coefficient(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    params.models[0].turbulence_model_solver.modeling_constants = SpalartAllmarasModelConstants(
        C_w2=2.718
    )
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_SA_with_modified_C_w2.json",
    )

    params.models[0].turbulence_model_solver = KOmegaSST(
        absolute_tolerance=1e-8,
        linear_solver=LinearSolver(max_iterations=15),
    )
    params.models[0].turbulence_model_solver.modeling_constants = KOmegaSSTModelConstants(
        C_sigma_omega1=2.718
    )
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_SST_with_modified_C_sigma_omega1.json",
    )


##::  Test with local test cases
def test_xv15_bet_disk(
    create_steady_hover_param,
    create_steady_airplane_param,
    create_unsteady_hover_param,
    create_unsteady_hover_UDD_param,
):
    param = create_steady_hover_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_steady_hover.json"
    )

    param = create_steady_airplane_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_steady_airplane.json"
    )

    param = create_unsteady_hover_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_unsteady_hover.json"
    )

    param = create_unsteady_hover_UDD_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_unsteady_hover_UDD.json"
    )


def test_xv15_bet_disk_nested_rotation(
    create_nested_rotation_param, cylinder_inner, cylinder_middle
):
    param = create_nested_rotation_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_nested_rotation.json"
    )


def test_porous_media(
    create_porous_media_box_param,
    create_porous_media_volume_zone_param,
):
    param = create_porous_media_box_param
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_porous_media_box.json")

    param = create_porous_media_volume_zone_param
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_porous_media_volume_zone.json"
    )


def test_actuator_disk_translation(actuator_disk_create_param):
    param = actuator_disk_create_param
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_actuator_disk.json")


def test_conjugate_heat_transfer(
    create_conjugate_heat_transfer_param,
):
    param = create_conjugate_heat_transfer_param
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_CHT_three_cylinders.json", atol=1e-6
    )


def test_vortex_propagation(create_vortex_propagation_param):
    param = create_vortex_propagation_param
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_vortex_propagation.json")


def test_periodic_euler_vortex(create_periodic_euler_vortex_param):
    param = create_periodic_euler_vortex_param
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_periodic_euler_vortex.json"
    )


def test_om6wing_wall_model(create_om6wing_wall_model_param):
    param = create_om6wing_wall_model_param
    translate_and_compare(
        param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6wing_wall_model.json", atol=1e-12
    )


def test_symmetryBC(create_symmetryBC_param):
    param = create_symmetryBC_param
    translate_and_compare(param, mesh_unit=1.0 * u.m, ref_json_file="Flow360_symmetryBC.json")


def test_XV15HoverMRF(create_XV15HoverMRF_param, rotation_cylinder):
    param = create_XV15HoverMRF_param
    translate_and_compare(param, mesh_unit=1.0 * u.m, ref_json_file="Flow360_XV15HoverMRF.json")


def test_NestedCylindersSRF(create_NestedCylindersSRF_param, srf_cylinder):
    param = create_NestedCylindersSRF_param
    translate_and_compare(
        param, mesh_unit=1.0 * u.m, ref_json_file="Flow360_NestedCylindersSRF.json"
    )


def test_heatFluxCylinder(create_heat_flux_cylinder_param):
    param = create_heat_flux_cylinder_param
    translate_and_compare(param, mesh_unit=1.0 * u.m, ref_json_file="Flow360_heatFluxCylinder.json")


def test_plateASI(create_plateASI_param):
    param = create_plateASI_param
    translate_and_compare(param, mesh_unit=0.1016 * u.m, ref_json_file="Flow360_plateASI.json")


def test_TurbFlatPlate137x97_BoxTrip(create_turb_flat_plate_box_trip_param):
    param = create_turb_flat_plate_box_trip_param
    translate_and_compare(
        param, mesh_unit=1.0 * u.m, ref_json_file="Flow360_TurbFlatPlate137x97_BoxTrip.json"
    )


def test_2dcrm_tutorial(get_2dcrm_tutorial_param):
    param = get_2dcrm_tutorial_param
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_tutorial_2dcrm.json")


def test_2dcrm_tutorial_temperature_c(get_2dcrm_tutorial_param_deg_c):
    param = get_2dcrm_tutorial_param_deg_c
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_tutorial_2dcrm.json")


def test_2dcrm_tutorial_temperature_f(get_2dcrm_tutorial_param_deg_f):
    param = get_2dcrm_tutorial_param_deg_f
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_tutorial_2dcrm.json")


def test_operating_condition(get_2dcrm_tutorial_param):
    converted = get_2dcrm_tutorial_param._preprocess(mesh_unit=1 * u.m)
    assertions.assertAlmostEqual(converted.operating_condition.velocity_magnitude.value, 0.2)
    assertions.assertAlmostEqual(
        converted.operating_condition.thermal_state.dynamic_viscosity.value,
        4.0121618e-08,
    )
    assertions.assertEqual(converted.operating_condition.thermal_state.temperature, 272.1 * u.K)
    assertions.assertAlmostEqual(
        converted.operating_condition.thermal_state.material.dynamic_viscosity.reference_viscosity.value,
        4.0121618e-08,
    )
    assertions.assertEqual(
        converted.operating_condition.thermal_state.material.dynamic_viscosity.effective_temperature,
        110.4 * u.K,
    )
    assertions.assertEqual(
        converted.operating_condition.thermal_state.material.get_dynamic_viscosity(
            converted.operating_condition.thermal_state.temperature
        ),
        4e-8,
    )


def test_initial_condition_and_restart():
    # 1. Default case
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
            ),
            models=[
                Fluid(
                    initial_condition=NavierStokesInitialCondition(
                        constants={"not_used": "-1.1"}, rho="rho", u="u", v="v", w="w", p="p"
                    )
                )
            ],
        )
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_initial_condition_v2.json"
    )

    # 2. Restart Manipulation
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
            ),
            models=[
                Fluid(
                    initial_condition=NavierStokesModifiedRestartSolution(
                        rho="rho*factor", u="u", v="v", w="w", p="p", constants={"factor": "1.1"}
                    )
                )
            ],
        )
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_restart_manipulation_v2.json", debug=True
    )


def test_user_defined_field():
    # 1. Default case
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
            ),
            models=[
                Fluid(
                    initial_condition=NavierStokesInitialCondition(
                        constants={"not_used": "-1.1"}, rho="rho", u="u", v="v", w="w", p="p"
                    )
                )
            ],
            user_defined_fields=[UserDefinedField(name="CpT", expression="C-p*T")],
        )
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_udf.json")

    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
            ),
            outputs=[
                VolumeOutput(
                    name="output",
                    output_fields=[
                        solution.Mach,
                        solution.velocity,
                        UserVariable(name="uuu", value=solution.velocity),
                    ],
                )
            ],
        )
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_expression_udf.json")


def test_boundaries():
    operating_condition = AerospaceCondition.from_mach(
        mach=0.84,
    )
    mass_flow_rate = (
        0.2
        * operating_condition.thermal_state.speed_of_sound
        * 1
        * u.m
        * u.m
        * operating_condition.thermal_state.density
    )
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=operating_condition,
            models=[
                Inflow(
                    name="inflow-1",
                    total_temperature=300 * u.K,
                    surfaces=Surface(name="boundary_name_A"),
                    spec=TotalPressure(
                        value=operating_condition.thermal_state.pressure * 0.9,
                    ),
                    velocity_direction=(1, 0, 0),
                ),
                Inflow(
                    name="inflow-2",
                    total_temperature=300 * u.K,
                    surfaces=Surface(name="boundary_name_B"),
                    spec=MassFlowRate(value=mass_flow_rate, ramp_steps=10),
                    velocity_direction=(0, 0, 1),
                ),
                Outflow(
                    name="outflow-1",
                    surfaces=Surface(name="boundary_name_E"),
                    spec=Pressure(operating_condition.thermal_state.pressure * 0.9),
                ),
                Outflow(
                    name="outflow-2",
                    surfaces=Surface(name="boundary_name_H"),
                    spec=MassFlowRate(value=mass_flow_rate, ramp_steps=10),
                ),
                Outflow(
                    name="outflow-3",
                    surfaces=Surface(name="boundary_name_F"),
                    spec=Mach(0.3),
                ),
                Wall(
                    name="slater-porous-bleed",
                    surfaces=Surface(name="boundary_name_G"),
                    velocity=SlaterPorousBleed(
                        static_pressure=12.0 * u.psi, porosity=0.49, activation_step=20
                    ),
                ),
                Wall(
                    surfaces=Surface(name="boundary_name_I"),
                    velocity=WallRotation(
                        axis=(0, 0, 1), center=(1, 2, 3) * u.m, angular_velocity=100 * u.rpm
                    ),
                ),
            ],
        )
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_boundaries.json")


def test_liquid_simulation_translation():
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=10 * u.m / u.s,
                alpha=5 * u.deg,
                beta=2 * u.deg,
                material=Water(name="my_water", density=1.1 * 10**3 * u.kg / u.m**3),
            ),
            models=[
                Fluid(
                    navier_stokes_solver=NavierStokesSolver(
                        low_mach_preconditioner=True,
                    )
                ),
                Wall(entities=Surface(name="fluid/body")),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
        )
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_liquid.json")

    with SI_unit_system:
        param = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=10 * u.m / u.s,
                alpha=5 * u.deg,
                beta=2 * u.deg,
                material=Water(name="my_water", density=1.1 * 10**3 * u.kg / u.m**3),
            ),
            models=[
                Wall(entities=Surface(name="fluid/body")),
                Freestream(entities=Surface(name="fluid/farfield")),
                Rotation(
                    volumes=[
                        GenericVolume(name="zone_zone_1", axis=[3, 4, 0], center=(1, 1, 1) * u.cm)
                    ],
                    spec=AngleExpression(
                        "-180/pi * atan(2 * 3.00 * 20.00 * 2.00/180*pi * "
                        "cos(2.00/180*pi * sin(0.05877271 * t_seconds)) * cos(0.05877271 * t_seconds) / 200.00) +"
                        " 2 * 2.00 * sin(0.05877271 * t_seconds) - 2.00 * sin(0.05877271 * t_seconds)"
                    ),
                    rotating_reference_frame_model=False,
                ),
            ],
            time_stepping=Unsteady(steps=100, step_size=0.4),
        )
        # Derivation:
        # Solver speed of sound = 10m/s / 0.05 = 200m/s
        # Flow360 time to seconds = 1m/(200m/s) = 0.005 s
        # t_seconds = (0.005 s * t)
    translate_and_compare(param, mesh_unit=1 * u.m, ref_json_file="Flow360_liquid_rotation_dd.json")


import flow360.component.simulation.user_code.core.context as context


@pytest.fixture()
def reset_context():
    context.default_context._values = {
        name: item for (name, item) in context.default_context._values.items() if "." in name
    }


def test_param_with_user_variables():
    some_dependent_variable_a = UserVariable(
        name="some_dependent_variable_a", value=[1.0 * u.m / u.s, 2.0 * u.m / u.s, 3.0 * u.m / u.s]
    )
    my_var = UserVariable(
        name="my_var", value=math.cross(some_dependent_variable_a, solution.velocity)
    )
    my_time_stepping_var = UserVariable(name="my_time_stepping_var", value=1.0 * u.s)
    my_temperature = UserVariable(
        name="my_temperature", value=(solution.temperature + (-10 * u.K)) * 1.8
    )
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=10 * u.m / u.s,
                alpha=5 * u.deg,
                beta=2 * u.deg,
                material=Water(name="my_water", density=1.000 * 10**3 * u.kg / u.m**3),
            ),
            models=[
                Wall(entities=Surface(name="fluid/body")),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            outputs=[
                VolumeOutput(
                    name="output",
                    output_fields=[
                        solution.Mach,
                        solution.velocity,
                        UserVariable(name="uuu", value=solution.velocity).in_units(
                            new_unit="km/ms"
                        ),
                        my_var,
                        my_temperature,
                    ],
                )
            ],
            time_stepping=Unsteady(step_size=my_time_stepping_var + 0.5 * u.s, steps=123),
        )
    # Mimicking real workflow where the Param is serialized and then deserialized
    params_validated, _, _ = validate_model(
        params_as_dict=param.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )

    assert params_validated
    translate_and_compare(
        params_validated,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_user_variable.json",
    )

    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(
                mach=0.84,
            ),
            models=[
                Solid(
                    volumes=[GenericVolume(name="CHTSolid")],
                    material=aluminum,
                    volumetric_heat_source="0",
                    initial_condition=HeatEquationInitialCondition(temperature="10"),
                ),
            ],
            outputs=[
                VolumeOutput(
                    name="output_heat",
                    output_fields=[
                        my_temperature,
                    ],
                )
            ],
            time_stepping=Unsteady(step_size=my_time_stepping_var + 0.5 * u.s, steps=123),
        )

    params_validated, _, _ = validate_model(
        params_as_dict=param.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )

    assert params_validated
    translate_and_compare(
        params_validated,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_user_variable_heat.json",
    )
