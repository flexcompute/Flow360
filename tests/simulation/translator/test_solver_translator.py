import json
import os
import unittest

import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    UserDefinedFarfield,
)
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
    Periodic,
    Pressure,
    SlaterPorousBleed,
    SlipWall,
    Supersonic,
    TotalPressure,
    Translational,
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
    PorousMedium,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    LiquidOperatingCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.outputs.outputs import (
    Isosurface,
    IsosurfaceOutput,
    MovingStatistic,
    ProbeOutput,
    SliceOutput,
    StreamlineOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    TimeAverageStreamlineOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    CustomVolume,
    GenericVolume,
    ImportedSurface,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.run_control.run_control import RunControl
from flow360.component.simulation.run_control.stopping_criterion import (
    StoppingCriterion,
)
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from flow360.component.simulation.utils import model_attribute_unlock
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
from tests.simulation.translator.utils.porousJump_params_generator import (
    create_porous_jump_zone_param,
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

import flow360.component.simulation.user_code.core.context as context
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    HeatEquationInitialCondition,
    Rotation,
    Solid,
)
from flow360.component.simulation.primitives import GenericVolume
from flow360.component.simulation.time_stepping.time_stepping import Unsteady


@pytest.fixture(autouse=True)
def reset_context():
    clear_context()


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
        print("=== translated ===\n", json.dumps(translated, indent=4, sort_keys=True))
        print("=== ref_dict ===\n", json.dumps(ref_dict, indent=4, sort_keys=True))
    assert compare_values(ref_dict, translated, atol=atol, rtol=rtol)


def test_om6wing_tutorial(get_om6Wing_tutorial_param):
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6Wing.json",
        debug=True,
    )


def test_porous_jump(create_porous_jump_zone_param):
    translate_and_compare(
        create_porous_jump_zone_param,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_porous_jump.json",
        debug=True,
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


def test_om6wing_streamlines(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    with SI_unit_system:
        streamlineOutput = StreamlineOutput(
            entities=[
                Point(name="point_streamline", location=(0.0, 1.0, 0.04)),
                PointArray(
                    name="pointarray_streamline",
                    start=(0.0, 0.0, 0.2),
                    end=(0.0, 1.0, 0.2),
                    number_of_points=20,
                ),
                PointArray2D(
                    name="pointarray2d_streamline",
                    origin=(0.0, 0.0, -0.2),
                    u_axis_vector=(0.0, 1.4, 0.0),
                    v_axis_vector=(0.0, 0.0, 0.4),
                    u_number_of_points=10,
                    v_number_of_points=10,
                ),
            ],
            output_fields=[solution.Cp, solution.velocity],
        )
    params.outputs.append(streamlineOutput)

    translate_and_compare(
        params,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_streamlines.json",
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


def test_om6wing_with_stopping_criterion_and_moving_statistic(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    monitored_variable = UserVariable(
        name="Helicity",
        value=math.dot(solution.velocity, solution.vorticity),
    )
    probe_output = ProbeOutput(
        name="point_legacy1",
        output_fields=[
            monitored_variable,
        ],
        probe_points=Point(
            name="Point1", location=(-0.026642, 0.56614, 0) * u.m, private_attribute_id="Point1"
        ),
        moving_statistic=MovingStatistic(method="mean", moving_window_size=200),
        private_attribute_id="11111",
    )
    criterion = StoppingCriterion(
        name="Criterion_Helicity",
        tolerance=18.66 * u.m / u.s**2,
        monitor_output=probe_output,
        monitor_field=monitored_variable,
    )
    params.run_control = RunControl(stopping_criteria=[criterion])
    params.outputs.append(probe_output)
    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_stopping_criterion_and_moving_statistic.json",
        debug=True,
    )


def test_stopping_criterion_tolerance_in_unit_system():
    """
    [Frontend] Test that an StoppingCriterion with the unit system as
    tolerance's units can be validated and translated.
    """

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "simulation_stopping_criterion.json"
        )
    ) as fp:
        params_as_dict = json.load(fp=fp)
    params_validated, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level="Case",
    )
    assert not errors, print(">>>", errors)
    assert params_validated.run_control.stopping_criteria[0].tolerance == 18.66 * u.m / u.s**2

    translate_and_compare(
        params_validated,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6wing_stopping_criterion_and_moving_statistic.json",
        debug=True,
    )


def test_om6wing_with_low_reynolds_correction(get_om6Wing_tutorial_param):
    params = get_om6Wing_tutorial_param
    params.models[0].turbulence_model_solver.low_reynolds_correction = True

    translate_and_compare(
        get_om6Wing_tutorial_param,
        mesh_unit=0.8059 * u.m,
        ref_json_file="Flow360_om6Wing_SA_with_low_reynolds_correction.json",
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
        converted.operating_condition.thermal_state.dynamic_viscosity.in_base(
            get_2dcrm_tutorial_param.flow360_unit_system
        ).value,
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
    assertions.assertAlmostEqual(
        converted.operating_condition.thermal_state.material.get_dynamic_viscosity(
            converted.operating_condition.thermal_state.temperature
        )
        .in_base(get_2dcrm_tutorial_param.flow360_unit_system)
        .value,
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
                Inflow(
                    name="inflow-3",
                    total_temperature=300 * u.K,
                    surfaces=Surface(name="boundary_name_C"),
                    spec=Supersonic(
                        total_pressure=operating_condition.thermal_state.pressure * 8.0,
                        static_pressure=operating_condition.thermal_state.pressure * 0.9,
                    ),
                    velocity_direction=(0, 1, 0),
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
            outputs=[
                VolumeOutput(
                    name="output",
                    output_fields=[solution.velocity],
                )
            ],
        )
        # Derivation:
        # Solver speed of sound = 10m/s / 0.05 = 200m/s
        # Flow360 time to seconds = 1m/(200m/s) = 0.005 s
        # t_seconds = (0.005 s * t)
    translate_and_compare(
        param, mesh_unit=1 * u.m, ref_json_file="Flow360_liquid_rotation_dd.json", debug=True
    )

    with model_attribute_unlock(param.operating_condition, "reference_velocity_magnitude"):
        param.operating_condition.reference_velocity_magnitude = 20 * u.m / u.s
    translate_and_compare(
        param,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_liquid_rotation_dd_with_ref_vel.json",
        debug=True,
    )


def test_param_with_user_variables():
    some_dependent_variable_a = UserVariable(
        name="some_dependent_variable_a", value=[1.0 * u.m / u.s, 2.0 * u.m / u.s, 3.0 * u.m / u.s]
    )
    cross_res = UserVariable(
        name="cross_res", value=math.cross(some_dependent_variable_a, solution.velocity)
    )
    dot_res = UserVariable(
        name="dot_res", value=math.dot(some_dependent_variable_a, solution.velocity)
    )
    magnitude_res = UserVariable(name="magnitude_res", value=math.magnitude(solution.velocity))
    add_res = UserVariable(
        name="add_res", value=math.add(some_dependent_variable_a, solution.velocity)
    )
    subtract_res = UserVariable(
        name="subtract_res", value=math.subtract(some_dependent_variable_a, solution.velocity)
    )
    sqrt_res = UserVariable(name="sqrt_res", value=math.sqrt(solution.velocity[2]))
    power_res = UserVariable(name="power_res", value=solution.velocity[1] ** 1.5)
    log_res = UserVariable(name="log_res", value=math.log(solution.Mach))
    exp_res = UserVariable(name="exp_res", value=math.exp(solution.CfVec[0]))
    abs_res = UserVariable(name="abs_res", value=math.abs(solution.velocity[0]) * np.pi)
    sin_float_res = UserVariable(name="sin_float_res", value=math.sin(solution.CfVec[0] * np.pi))
    cos_deg_res = UserVariable(
        name="cos_deg_res", value=math.cos(solution.CfVec[1] * np.pi * u.deg)
    )
    tan_rad_res = UserVariable(
        name="tan_rad_res", value=math.tan(solution.CfVec[2] * np.pi * u.rad)
    )
    asin_res = UserVariable(name="asin_res", value=math.asin(solution.mut_ratio))
    acos_res = UserVariable(name="acos_res", value=math.acos(solution.Cp))
    atan_res = UserVariable(name="atan_res", value=math.atan(solution.Cpt))
    just_atan_auto = UserVariable(name="just_atan_auto", value=math.atan(solution.Cpt_auto))
    min_res = UserVariable(
        name="min_res", value=math.min(solution.vorticity[2], solution.vorticity[1])
    )
    max_res = UserVariable(
        name="max_res", value=math.max(solution.vorticity[0], solution.vorticity[1])
    )
    my_time_stepping_var = UserVariable(name="my_time_stepping_var", value=1.0 * u.s)
    const_value = UserVariable(name="const_value", value=1.0 * u.m / u.s)
    const_value_dimensionless = UserVariable(name="const_value_dimensionless", value=1.123)
    const_array = UserVariable(
        name="const_array", value=[1.0 * u.m / u.s, 2.0 * u.m / u.s, 3.0 * u.m / u.s]
    )
    const_array_dimensionless = UserVariable(
        name="const_array_dimensionless", value=[1.0, 2.0, 3.0]
    )
    my_temperature = UserVariable(
        name="my_temperature", value=(solution.temperature + (-10 * u.K)) * 1.8
    )
    surface_integral_variable = UserVariable(
        name="MassFluxProjected",
        value=-1 * solution.density * math.dot(solution.velocity, solution.node_unit_normal),
    )
    iso_field_pressure = UserVariable(
        name="iso_field_pressure",
        value=0.5 * solution.Cp * solution.density * math.magnitude(solution.velocity) ** 2,
    )
    iso1 = Isosurface(
        name="iso_pressure",
        field=iso_field_pressure,
        iso_value=10 * u.Pa,
        wall_distance_clip_threshold=0.1 * u.m,
    )
    iso_field_random_units = UserVariable(
        name="iso_field_random_units", value=solution.velocity[0] * 2 * u.lb
    )
    iso2 = Isosurface(
        name="iso_field_random_units", field=iso_field_random_units, iso_value=10 * u.lb * u.m / u.s
    )
    iso_field_velocity = UserVariable(
        name="iso_field_velocity_mag",
        value=math.magnitude(solution.velocity),
    )
    iso3 = Isosurface(
        name="iso_surf_velocity_mag", field=iso_field_velocity, iso_value=10 * u.m / u.s
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
                        const_value,
                        const_value_dimensionless,
                        const_array,
                        const_array_dimensionless,
                        my_temperature,
                        cross_res,
                        dot_res,
                        add_res,
                        magnitude_res,
                        subtract_res,
                        sqrt_res,
                        log_res,
                        power_res,
                        abs_res,
                        asin_res,
                        acos_res,
                        atan_res,
                        min_res,
                        max_res,
                        just_atan_auto,
                    ],
                ),
                IsosurfaceOutput(
                    name="iso_pressure",
                    entities=[iso1],
                    output_fields=[
                        UserVariable(name="ppp", value=solution.pressure).in_units(new_unit="psf"),
                    ],
                ),
                IsosurfaceOutput(
                    name="iso_random",
                    entities=[iso2],
                    output_fields=[
                        UserVariable(
                            name="velocity_km_per_hr", value=solution.velocity[0]
                        ).in_units(new_unit="km/hr"),
                    ],
                ),
                IsosurfaceOutput(
                    name="iso_velocity_mag",
                    entities=[iso3],
                    output_fields=[
                        UserVariable(name="velocity_mile_per_hr", value=solution.velocity).in_units(
                            new_unit="mile/hr"
                        ),
                    ],
                ),
                SurfaceIntegralOutput(
                    name="MassFluxIntegral",
                    output_fields=[surface_integral_variable],
                    entities=Surface(name="VOLUME/LEFT"),
                ),
                SurfaceIntegralOutput(
                    name="MassFluxIntegralImported",
                    output_fields=[surface_integral_variable],
                    entities=ImportedSurface(name="imported", file_name="imported.stl"),
                ),
                SurfaceOutput(
                    name="surface_output",
                    entities=Surface(name="fluid/body"),
                    output_fields=[
                        exp_res,
                        sin_float_res,
                        cos_deg_res,
                        tan_rad_res,
                    ],
                ),
            ],
            time_stepping=Unsteady(step_size=my_time_stepping_var + 0.5 * u.s, steps=123),
        )
    # Mimicking real workflow where the Param is serialized and then deserialized
    params_validated, errors, _ = validate_model(
        params_as_dict=param.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )
    assert not errors, print(">>>", errors)

    translated = get_solver_json(params_validated, mesh_unit=1 * u.m)
    units = iso_field_random_units.value.get_output_units(input_params=params_validated)
    assert units == u.kg * u.m / u.s
    assert (
        translated["isoSurfaceOutput"]["isoSurfaces"]["iso_field_random_units"][
            "surfaceFieldMagnitude"
        ]
        == params_validated.outputs[2].entities.items[0].iso_value.to(units).v.item()
    )

    assert params_validated
    translate_and_compare(
        params_validated, mesh_unit=1 * u.m, ref_json_file="Flow360_user_variable.json", debug=True
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


def test_isosurface_iso_value_in_unit_system():
    """
    [Frontend] Test that an Isosurface with the unit system as
    iso_value's units can be validated and translated.
    """

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "simulation_isosurface.json"
        )
    ) as fp:
        params_as_dict = json.load(fp=fp)
    params_validated, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level="Case",
    )
    assert not errors, print(">>>", errors)
    assert params_validated.outputs[0].entities.items[0].iso_value == 3000 * u.Pa
    assert params_validated.outputs[1].entities.items[0].iso_value == 45.359237 * u.cm * u.g / u.s
    assert params_validated.outputs[2].entities.items[0].iso_value == 2125 * u.psf
    assert params_validated.outputs[3].entities.items[0].iso_value == 0.5 * u.dimensionless

    translate_and_compare(
        params_validated,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_user_variable_isosurface.json",
        debug=True,
    )

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "simulation_isosurface.json"
        )
    ) as fp:
        params_as_dict = json.load(fp=fp)

    params_as_dict["outputs"][2]["entities"]["items"][0]["field"]["name"] = "uuu"
    params_validated, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.SERVICE,
        root_item_type="Case",
        validation_level="Case",
    )
    assert len(errors) == 2
    assert errors[0]["loc"] == ("outputs", 2, "entities", "items", 0, "field")
    assert (
        errors[0]["msg"]
        == "Value error, The isosurface field (uuu) must be defined with a scalar variable."
    )
    assert errors[1]["loc"] == ("outputs", 2, "entities", "items", 0, "iso_value")
    assert (
        errors[1]["msg"]
        == "Value error, The isosurface field is invalid and therefore unit inference is not possible."
    )


class TestHashingRobustness:
    @pytest.fixture
    def reference_hash(self):
        """Create a reference hash from the first boundary order configuration."""
        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=[
                    Wall(entities=[Surface(name="fluid/wall1"), Surface(name="fluid/wall2")]),
                    Freestream(entities=Surface(name="fluid/farfield")),
                ],
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        return SimulationParams._calculate_hash(translated)

    @pytest.mark.parametrize(
        "boundary_order",
        [
            # Test case 1: Single Wall with multiple surfaces (reference case)
            [
                Wall(entities=[Surface(name="fluid/wall1"), Surface(name="fluid/wall2")]),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            # Test case 2: Multiple Wall models with single surfaces each
            [
                Wall(entities=[Surface(name="fluid/wall2")]),
                Wall(entities=[Surface(name="fluid/wall1")]),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            # Test case 3: Different order of Wall models
            [
                Wall(entities=[Surface(name="fluid/wall1")]),
                Wall(entities=[Surface(name="fluid/wall2")]),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            # Test case 4: Freestream first, then Walls
            [
                Freestream(entities=Surface(name="fluid/farfield")),
                Wall(entities=[Surface(name="fluid/wall1"), Surface(name="fluid/wall2")]),
            ],
        ],
    )
    def test_diff_boundary_order(self, boundary_order, reference_hash):
        """Test that different boundary condition orders produce the same hash."""
        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=boundary_order,
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        hash_value = SimulationParams._calculate_hash(translated)

        # All test cases should produce the same hash as the reference
        assert hash_value == reference_hash

    @pytest.fixture
    def udf_reference_hash(self):
        """Create a reference hash from the first UDF order configuration."""
        from flow360.component.simulation.services import clear_context

        clear_context()
        var_1 = UserVariable(name="uuu", value=solution.velocity[0] + 123 * u.km / u.s)
        var_2 = UserVariable(name="vvv", value=solution.velocity[1] - 234 * u.mm / u.week)
        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=[
                    Wall(entities=[Surface(name="fluid/wall1"), Surface(name="fluid/wall2")]),
                    Freestream(entities=Surface(name="fluid/farfield")),
                ],
                outputs=[
                    SurfaceOutput(
                        name="surface_output",
                        entities=Surface(name="fluid/wall1"),
                        output_fields=[var_1, var_2],
                    )
                ],
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        return SimulationParams._calculate_hash(translated)

    @pytest.mark.parametrize(
        "udf_order",
        [
            # Test case 1: var_1, var_2 (reference order)
            ["var_1", "var_2"],
            # Test case 2: var_2, var_1 (reversed order)
            ["var_2", "var_1"],
        ],
    )
    def test_different_UDF_order(self, udf_order, udf_reference_hash):
        """Test that different UDF field orders produce the same hash."""
        from flow360.component.simulation.services import clear_context

        clear_context()
        var_1 = UserVariable(name="uuu", value=solution.velocity[0] + 123 * u.km / u.s)
        var_2 = UserVariable(name="vvv", value=solution.velocity[1] - 234 * u.mm / u.week)

        # Create output fields based on the order parameter
        output_fields = [var_1 if field == "var_1" else var_2 for field in udf_order]

        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=[
                    Wall(entities=[Surface(name="fluid/wall1"), Surface(name="fluid/wall2")]),
                    Freestream(entities=Surface(name="fluid/farfield")),
                ],
                outputs=[
                    SurfaceOutput(
                        name="surface_output",
                        entities=Surface(name="fluid/wall1"),
                        output_fields=output_fields,
                    )
                ],
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        hash_value = SimulationParams._calculate_hash(translated)

        # All test cases should produce the same hash as the reference
        assert hash_value == udf_reference_hash

    @pytest.fixture
    def udd_reference_hash(self):
        """Create a reference hash from the first entity order configuration."""
        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=[
                    Inflow(
                        entities=[Surface(name="fluid/in1"), Surface(name="fluid/in2")],
                        spec=MassFlowRate(value=110),
                        total_temperature=288.15 * u.K,
                    ),
                    Outflow(
                        entities=[Surface(name="fluid/out1"), Surface(name="fluid/out2")],
                        spec=MassFlowRate(value=120),
                    ),
                ],
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        return SimulationParams._calculate_hash(translated)

    @pytest.mark.parametrize(
        "entity_order",
        [
            # Test case 1: Original order (reference case)
            {
                "inflow_entities": [Surface(name="fluid/in1"), Surface(name="fluid/in2")],
                "outflow_entities": [Surface(name="fluid/out1"), Surface(name="fluid/out2")],
            },
            # Test case 2: Reversed order
            {
                "inflow_entities": [Surface(name="fluid/in2"), Surface(name="fluid/in1")],
                "outflow_entities": [Surface(name="fluid/out1"), Surface(name="fluid/out2")],
            },
        ],
    )
    def test_different_UDD_ordering_by_inflow_outflow(self, entity_order, udd_reference_hash):
        """Test that different entity orders in inflow/outflow boundaries produce the same hash."""
        with SI_unit_system:
            param = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=10),
                models=[
                    Inflow(
                        entities=entity_order["inflow_entities"],
                        spec=MassFlowRate(value=110),
                        total_temperature=288.15 * u.K,
                    ),
                    Outflow(
                        entities=entity_order["outflow_entities"],
                        spec=MassFlowRate(value=120),
                    ),
                ],
            )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        hash_value = SimulationParams._calculate_hash(translated)

        # All test cases should produce the same hash as the reference
        assert hash_value == udd_reference_hash


def test_auto_ref_area_settings():
    """
    [Frontend] Test that the auto reference area settings are translated correctly.
    """
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "simulation_with_auto_area.json"
        )
    ) as fp:
        params_as_dict = json.load(fp=fp)

    params, err, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    assert not err, print(">>>", err)
    translated = get_solver_json(params, mesh_unit=1 * u.m)

    assert compare_values(
        {
            "refArea": 0.0040039062500000005,
            "momentCenter": [0.0, 0.0, 0.0],
            "momentLength": [0.01, 0.01, 0.010001],
        },
        translated["geometry"],
    )


def test_custom_volume_translation():
    zone_2 = CustomVolume(name="zone2", boundaries=[Surface(name="face2")])
    zone_2.axes = [(1, 0, 0), (0, 1, 0)]

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    planar_face_tolerance=1e-4,
                ),
                volume_zones=[
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(name="zone1", boundaries=[Surface(name="face1")]),
                            zone_2,
                        ],
                    ),
                    UserDefinedFarfield(),
                ],
            ),
            operating_condition=AerospaceCondition(velocity_magnitude=10),
            models=[
                PorousMedium(
                    entities=[zone_2],
                    darcy_coefficient=(1, 0, 0) / u.m**2,
                    forchheimer_coefficient=(1, 0, 0) / u.m,
                    volumetric_heat_source=1.0 * u.W / u.m**3,
                ),
            ],
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=True,
                project_entity_info=SurfaceMeshEntityInfo(
                    boundaries=[
                        Surface(name="face1"),
                        Surface(name="face2"),
                    ]
                ),
            ),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="None",  # Skip validation for translation test
    )
    translated = get_solver_json(params, mesh_unit=1 * u.m)
    translate_and_compare(
        params,
        mesh_unit=1 * u.m,
        ref_json_file="Flow360_custom_volume_translation.json",
        debug=True,
    )


def test_ghost_periodic():
    geometry = Geometry.from_local_storage(
        geometry_id="geo-2f3c2143-436b-4a42-beab-aa191f49309c",  # placeholder UUID
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "ghost_periodic_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-2f3c2143-436b-4a42-beab-aa191f49309c",
                name="geometry_name_placeholder",
                cloud_path_prefix="s3_path_placeholder",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("groupByBodyId")  # manual grouping needed for from_local_storage
    far_field_zone = AutomatedFarfield(method="quasi-3d-periodic")
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_max_edge_length=0.1,
                    boundary_layer_growth_rate=1.2,
                    boundary_layer_first_layer_thickness=1e-6,
                ),
                volume_zones=[far_field_zone],
            ),
            reference_geometry=ReferenceGeometry(),
            operating_condition=AerospaceCondition(velocity_magnitude=10, alpha=0 * u.deg),
            time_stepping=Steady(max_steps=1000),
            models=[
                Wall(surfaces=[geometry["*"]]),
                Freestream(surfaces=[far_field_zone.farfield]),
                Periodic(surface_pairs=[far_field_zone.symmetry_planes], spec=Translational()),
            ],
            # Define output parameters for the simulation
            outputs=[
                SurfaceOutput(surfaces=geometry["*"], output_fields=["Cp", "Cf", "yPlus", "CfVec"])
            ],
        )
    processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, False, False)
    translate_and_compare(
        processed_params, mesh_unit=1 * u.m, ref_json_file="Flow360_ghost_periodic.json", debug=True
    )
