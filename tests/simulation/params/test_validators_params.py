import json
import re
import unittest
from typing import Literal

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.material import SolidMaterial, aluminum
from flow360.component.simulation.models.solver_numerics import (
    DetachedEddySimulation,
    TransitionModelSolver,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    Periodic,
    SlaterPorousBleed,
    SlipWall,
    Translational,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    Fluid,
    HeatEquationInitialCondition,
    NavierStokesInitialCondition,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.output_entities import (
    Isosurface,
    Point,
    Slice,
)
from flow360.component.simulation.outputs.outputs import (
    IsosurfaceOutput,
    ProbeOutput,
    SliceOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    TimeAverageSliceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericVolume,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.validation.validation_context import (
    ALL,
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
    ValidationLevelContext,
)

assertions = unittest.TestCase("__init__")


@pytest.fixture()
def surface_output_with_wall_metric():
    surface_output = SurfaceOutput(
        name="surface",
        surfaces=[Surface(name="noSlipWall")],
        write_single_file=True,
        output_fields=["wallFunctionMetric"],
    )
    return surface_output


@pytest.fixture()
def volume_output_with_SA_hybrid_model():
    volume_output = VolumeOutput(name="volume", output_fields=["SpalartAllmaras_hybridModel"])
    return volume_output


@pytest.fixture()
def volume_output_with_kOmega_hybrid_model():
    volume_output = VolumeOutput(name="volume", output_fields=["kOmegaSST_hybridModel"])
    return volume_output


@pytest.fixture()
def surface_output_with_low_mach_precond():
    surface_output = SurfaceOutput(
        name="surface",
        surfaces=[Surface(name="noSlipWall")],
        write_single_file=True,
        output_fields=["lowMachPreconditionerSensor"],
    )
    return surface_output


@pytest.fixture()
def surface_output_with_numerical_dissipation():
    surface_output = SurfaceOutput(
        name="surface",
        surfaces=[Surface(name="noSlipWall")],
        write_single_file=True,
        output_fields=["numericalDissipationFactor"],
    )
    return surface_output


@pytest.fixture()
def wall_model_with_function():
    wall_model = Wall(name="wall", surfaces=[Surface(name="noSlipWall")], use_wall_function=True)
    return wall_model


@pytest.fixture()
def wall_model_without_function():
    wall_model = Wall(name="wall", surfaces=[Surface(name="noSlipWall")], use_wall_function=False)
    return wall_model


@pytest.fixture()
def fluid_model_with_low_mach_precond():
    fluid_model = Fluid()
    fluid_model.navier_stokes_solver.low_mach_preconditioner = True
    return fluid_model


@pytest.fixture()
def fluid_model_with_low_numerical_dissipation():
    fluid_model = Fluid()
    fluid_model.navier_stokes_solver.numerical_dissipation_factor = 0.2
    return fluid_model


@pytest.fixture()
def fluid_model_with_hybrid_model():
    fluid_model = Fluid()
    fluid_model.turbulence_model_solver.hybrid_model = DetachedEddySimulation()
    return fluid_model


@pytest.fixture()
def fluid_model():
    fluid_model = Fluid()
    return fluid_model


def test_consistency_wall_function_validator(
    surface_output_with_wall_metric, wall_model_with_function, wall_model_without_function
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[wall_model_with_function], outputs=[surface_output_with_wall_metric]
        )

    assert params

    message = (
        "To use 'wallFunctionMetric' for output specify a Wall model with use_wall_function=true. "
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[wall_model_without_function], outputs=[surface_output_with_wall_metric]
        )


def test_consistency_wall_function_validator():

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Wall(velocity=["0.1*t", "0.2*t", "0.3*t"], surfaces=[Surface(name="noSlipWall")])
            ]
        )

    assert params

    message = "Using `SlaterPorousBleed` with wall function is not supported currently."

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[
                Wall(
                    velocity=SlaterPorousBleed(porosity=0.2, static_pressure=0.1),
                    surfaces=[Surface(name="noSlipWall")],
                    use_wall_function=True,
                )
            ]
        )


def test_low_mach_preconditioner_validator(
    surface_output_with_low_mach_precond, fluid_model_with_low_mach_precond, fluid_model
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_low_mach_precond],
            outputs=[surface_output_with_low_mach_precond],
        )

    assert params

    message = (
        "Low-Mach preconditioner output requested, but low_mach_preconditioner is not enabled. "
        "You can enable it via model.navier_stokes_solver.low_mach_preconditioner = True for a Fluid "
        "model in the models field of the simulation object."
    )

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(models=[fluid_model], outputs=[surface_output_with_low_mach_precond])


def test_numerical_dissipation_mode_validator(
    surface_output_with_numerical_dissipation,
    fluid_model_with_low_numerical_dissipation,
    fluid_model,
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_low_numerical_dissipation],
            outputs=[surface_output_with_numerical_dissipation],
        )

    assert params

    message = (
        "Numerical dissipation factor output requested, but low dissipation mode is not enabled. "
        "You can enable it via model.navier_stokes_solver.numerical_dissipation_factor = True for a Fluid "
        "model in the models field of the simulation object."
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model], outputs=[surface_output_with_numerical_dissipation]
        )


def test_hybrid_model_wall_function_validator(
    volume_output_with_SA_hybrid_model,
    volume_output_with_kOmega_hybrid_model,
    fluid_model_with_hybrid_model,
    fluid_model,
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_hybrid_model],
            outputs=[volume_output_with_SA_hybrid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )

    assert params

    message = "kOmegaSST_hybridModel output can only be specified with kOmegaSST turbulence model and hybrid RANS-LES used."

    # Invalid simulation params (wrong output type)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            models=[fluid_model_with_hybrid_model],
            outputs=[volume_output_with_kOmega_hybrid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )

    # Invalid simulation params (no hybrid)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            models=[fluid_model],
            outputs=[volume_output_with_kOmega_hybrid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )


def test_hybrid_model_for_unsteady_validator(
    fluid_model_with_hybrid_model,
):

    message = "hybrid RANS-LES model can only be used in unsteady simulations."

    # Invalid simulation params (using hybrid model for steady simulations)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(models=[fluid_model_with_hybrid_model])


def test_cht_solver_settings_validator(
    fluid_model,
):
    timestepping_unsteady = Unsteady(steps=12, step_size=0.1 * u.s)
    fluid_model_with_initial_condition = Fluid(
        initial_condition=NavierStokesInitialCondition(rho="1;", u="1;", v="1;", w="1;", p="1;")
    )
    solid_model = Solid(
        volumes=[GenericVolume(name="CHTSolid")],
        material=aluminum,
        volumetric_heat_source="0",
        initial_condition=HeatEquationInitialCondition(temperature="10"),
    )
    solid_model_without_initial_condition = Solid(
        volumes=[GenericVolume(name="CHTSolid")],
        material=aluminum,
        volumetric_heat_source="0",
    )
    solid_model_without_specific_heat_capacity = Solid(
        volumes=[GenericVolume(name="CHTSolid")],
        material=SolidMaterial(
            name="aluminum_without_specific_heat_capacity",
            thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
            density=2710 * u.kg / u.m**3,
        ),
        volumetric_heat_source="0",
        initial_condition=HeatEquationInitialCondition(temperature="10;"),
    )
    solid_model_without_density = Solid(
        volumes=[GenericVolume(name="CHTSolid")],
        material=SolidMaterial(
            name="aluminum_without_density",
            thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
            specific_heat_capacity=903 * u.m**2 / u.s**2 / u.K,
        ),
        volumetric_heat_source="0",
        initial_condition=HeatEquationInitialCondition(temperature="10;"),
    )
    surface_output_with_residual_heat_solver = SurfaceOutput(
        name="surface",
        surfaces=[Surface(name="noSlipWall")],
        write_single_file=True,
        output_fields=["residualHeatSolver"],
    )

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model, solid_model],
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
        )

    assert params

    message = (
        "Heat equation output variables: residualHeatSolver is requested in "
        f"{surface_output_with_residual_heat_solver.output_type} with no `Solid` model defined."
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model],
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
        )

    message = (
        "In `Solid` model -> material, both `specific_heat_capacity` and `density` "
        "need to be specified for unsteady simulations."
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model, solid_model_without_specific_heat_capacity],
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
        )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model, solid_model_without_density],
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
        )

    message = (
        "In `Solid` model, the initial condition needs to be specified for unsteady simulations."
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model, solid_model_without_initial_condition],
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
        )


def test_transition_model_solver_settings_validator():
    transition_model_solver = TransitionModelSolver()
    assert transition_model_solver

    with SI_unit_system:
        params = SimulationParams(
            models=[Fluid(transition_model_solver=transition_model_solver)],
        )
        assert params.models[0].transition_model_solver.N_crit == 8.15

    with pytest.raises(
        pd.ValidationError,
        match="N_crit and turbulence_intensity_percent cannot be specified at the same time.",
    ):
        transition_model_solver = TransitionModelSolver(
            update_jacobian_frequency=5,
            equation_evaluation_frequency=10,
            max_force_jac_update_physical_steps=10,
            order_of_accuracy=1,
            turbulence_intensity_percent=1.2,
            N_crit=2,
        )

    transition_model_solver = TransitionModelSolver(
        update_jacobian_frequency=5,
        equation_evaluation_frequency=10,
        max_force_jac_update_physical_steps=10,
        order_of_accuracy=1,
        turbulence_intensity_percent=1.2,
    )

    with SI_unit_system:
        params = SimulationParams(
            models=[Fluid(transition_model_solver=transition_model_solver)],
        )
        assert params.models[0].transition_model_solver.N_crit == 2.3598473252999543
        assert params.models[0].transition_model_solver.turbulence_intensity_percent is None


def test_incomplete_BC():
    ##:: Construct a dummy asset cache

    wall_1 = Surface(
        name="wall_1", private_attribute_is_interface=False, private_attribute_tag_key="test"
    )
    periodic_1 = Surface(
        name="periodic_1", private_attribute_is_interface=False, private_attribute_tag_key="test"
    )
    periodic_2 = Surface(
        name="periodic_2", private_attribute_is_interface=False, private_attribute_tag_key="test"
    )
    i_exist = Surface(
        name="i_exist", private_attribute_is_interface=False, private_attribute_tag_key="test"
    )
    no_bc = Surface(
        name="no_bc", private_attribute_is_interface=False, private_attribute_tag_key="test"
    )
    some_interface = Surface(
        name="some_interface", private_attribute_is_interface=True, private_attribute_tag_key="test"
    )

    asset_cache = AssetCache(
        project_length_unit="inch",
        project_entity_info=VolumeMeshEntityInfo(
            boundaries=[wall_1, periodic_1, periodic_2, i_exist, some_interface, no_bc]
        ),
    )
    auto_farfield = AutomatedFarfield(name="my_farfield")

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"The following boundaries do not have a boundary condition: no_bc. Please add them to a boundary condition model in the `models` section."
        ),
    ):
        with ValidationLevelContext(ALL):
            with SI_unit_system:
                SimulationParams(
                    meshing=MeshingParams(
                        defaults=MeshingDefaults(
                            boundary_layer_first_layer_thickness=1e-10,
                            surface_max_edge_length=1e-10,
                        )
                    ),
                    models=[
                        Fluid(),
                        Wall(entities=wall_1),
                        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                        SlipWall(entities=[i_exist]),
                        Freestream(entities=auto_farfield.farfield),
                    ],
                    private_attribute_asset_cache=asset_cache,
                )
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"The following boundaries are not known `Surface` entities but appear in the `models` section: plz_dont_do_this."
        ),
    ):
        with ValidationLevelContext(ALL):
            with SI_unit_system:
                SimulationParams(
                    meshing=MeshingParams(
                        defaults=MeshingDefaults(
                            boundary_layer_first_layer_thickness=1e-10,
                            surface_max_edge_length=1e-10,
                        )
                    ),
                    models=[
                        Fluid(),
                        Wall(entities=[wall_1]),
                        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                        SlipWall(entities=[i_exist]),
                        Freestream(entities=auto_farfield.farfield),
                        SlipWall(entities=[Surface(name="plz_dont_do_this"), no_bc]),
                    ],
                    private_attribute_asset_cache=asset_cache,
                )


def test_duplicate_entities_in_models():
    entity_generic_volume = GenericVolume(name="Duplicate Volume")
    entity_surface = Surface(name="Duplicate Surface")
    entity_cylinder = Cylinder(
        name="Duplicate Cylinder",
        outer_radius=1 * u.cm,
        height=1 * u.cm,
        center=(0, 0, 0) * u.cm,
        axis=(0, 0, 1),
        private_attribute_id="1",
    )
    entity_box = Box(
        name="Box",
        axis_of_rotation=(1, 0, 0),
        angle_of_rotation=45 * u.deg,
        center=(1, 1, 1) * u.m,
        size=(0.2, 0.3, 2) * u.m,
        private_attribute_id="2",
    )
    entity_box_same_name = Box(
        name="Box",
        axis_of_rotation=(1, 0, 0),
        angle_of_rotation=45 * u.deg,
        center=(1, 1, 1) * u.m,
        size=(0.2, 0.3, 2) * u.m,
        private_attribute_id="3",
    )
    volume_model1 = Solid(
        volumes=[entity_generic_volume],
        material=aluminum,
        volumetric_heat_source="0",
    )
    volume_model2 = volume_model1
    surface_model1 = SlipWall(entities=[entity_surface])
    surface_model2 = Wall(entities=[entity_surface])
    surface_model3 = surface_model1

    rotation_model1 = Rotation(
        volumes=[entity_cylinder],
        name="innerRotation",
        spec=AngleExpression("sin(t)"),
    )
    rotation_model2 = Rotation(
        volumes=[entity_cylinder],
        name="outerRotation",
        spec=AngleExpression("sin(2t)"),
    )
    porous_medium_model1 = PorousMedium(
        volumes=entity_box,
        darcy_coefficient=(1e6, 0, 0) / u.m**2,
        forchheimer_coefficient=(1, 0, 0) / u.m,
        volumetric_heat_source=1.0 * u.W / u.m**3,
    )
    porous_medium_model2 = PorousMedium(
        volumes=entity_box_same_name,
        darcy_coefficient=(3e5, 0, 0) / u.m**2,
        forchheimer_coefficient=(1, 0, 0) / u.m,
        volumetric_heat_source=1.0 * u.W / u.m**3,
    )

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[volume_model1, surface_model1],
        )

    assert params

    # Valid simulation params with the same Box name in the PorousMedium model
    with SI_unit_system:
        params = SimulationParams(
            models=[porous_medium_model1, porous_medium_model2],
        )

    assert params

    message = (
        f"Surface entity `{entity_surface.name}` appears multiple times in `{surface_model1.type}`, `{surface_model2.type}` models.\n"
        f"Volume entity `{entity_generic_volume.name}` appears multiple times in `{volume_model1.type}` model.\n"
    )

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[volume_model1, volume_model2, surface_model1, surface_model2, surface_model3],
        )

    message = f"Volume entity `{entity_cylinder.name}` appears multiple times in `{rotation_model1.type}` model.\n"

    # Invalid simulation params (Draft Entity)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[rotation_model1, rotation_model2],
        )


def test_valid_reference_velocity():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Reference velocity magnitude/Mach must be provided when freestream velocity magnitude/Mach is 0."
        ),
    ):
        with SI_unit_system:
            SimulationParams(operating_condition=AerospaceCondition(velocity_magnitude=0))


def test_output_fields_with_user_defined_fields():
    surface_1 = Surface(name="some_random_surface")
    # 1: No user defined fields
    msg = "In `outputs`[0] SurfaceOutput:, not_valid_field is not a valid output field name. Allowed fields are ['Cp', 'Cpt', 'gradW', 'kOmega', 'Mach', 'mut', 'mutRatio', 'nuHat', 'primitiveVars', 'qcriterion', 'residualNavierStokes', 'residualTransition', 'residualTurbulence', 's', 'solutionNavierStokes', 'solutionTransition', 'solutionTurbulence', 'T', 'vorticity', 'vorticityMagnitude', 'wallDistance', 'numericalDissipationFactor', 'residualHeatSolver', 'VelocityRelative', 'lowMachPreconditionerSensor', 'CfVec', 'Cf', 'heatFlux', 'nodeNormals', 'nodeForcesPerUnitArea', 'yPlus', 'wallFunctionMetric', 'heatTransferCoefficientStaticTemperature', 'heatTransferCoefficientTotalTemperature']."
    with pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            _ = SimulationParams(
                outputs=[
                    SurfaceOutput(
                        name="surface", output_fields=["not_valid_field"], entities=[surface_1]
                    )
                ],
            )

    # 2: User defined fields
    with SI_unit_system:
        _ = SimulationParams(
            outputs=[VolumeOutput(name="vo", output_fields=["not_valid_field"])],
            user_defined_fields=[
                UserDefinedField(name="not_valid_field", expression="primitiveVars[0] *3.1415926")
            ],
        )

    msg = "In `outputs`[1] SliceOutput:, not_valid_field_2 is not a valid output field name. Allowed fields are ['Cp', 'Cpt', 'gradW', 'kOmega', 'Mach', 'mut', 'mutRatio', 'nuHat', 'primitiveVars', 'qcriterion', 'residualNavierStokes', 'residualTransition', 'residualTurbulence', 's', 'solutionNavierStokes', 'solutionTransition', 'solutionTurbulence', 'T', 'vorticity', 'vorticityMagnitude', 'wallDistance', 'numericalDissipationFactor', 'residualHeatSolver', 'VelocityRelative', 'lowMachPreconditionerSensor', 'betMetrics', 'betMetricsPerDisk', 'linearResidualNavierStokes', 'linearResidualTurbulence', 'linearResidualTransition', 'SpalartAllmaras_hybridModel', 'kOmegaSST_hybridModel', 'localCFL', 'not_valid_field']."
    with pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            _ = SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="po",
                        output_fields=["not_valid_field"],
                        probe_points=[Point(name="pt1", location=(1, 2, 3))],
                    ),
                    SliceOutput(
                        name="so",
                        output_fields=["not_valid_field_2"],
                        slices=[Slice(name="slice", normal=(1, 2, 3), origin=(0, 0, 0))],
                    ),
                ],
                user_defined_fields=[
                    UserDefinedField(
                        name="not_valid_field", expression="primitiveVars[0] *3.1415926"
                    )
                ],
            )

    msg = "In `outputs`[0] SurfaceIntegralOutput:, Cp is not a valid output field name. Allowed fields are ['not_valid_field']."
    with pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            _ = SimulationParams(
                outputs=[
                    SurfaceIntegralOutput(
                        name="po",
                        output_fields=["Cp"],
                        surfaces=[surface_1],
                    )
                ],
                user_defined_fields=[
                    UserDefinedField(
                        name="not_valid_field", expression="primitiveVars[0] *3.1415926"
                    )
                ],
            )

    msg = "`SurfaceIntegralOutput` can only be used with `UserDefinedField`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            _ = SimulationParams(
                outputs=[
                    SurfaceIntegralOutput(
                        name="po",
                        output_fields=["Cp"],
                        surfaces=[surface_1],
                    )
                ]
            )

    msg = "In `outputs`[1] IsosurfaceOutput:, Cpp is not a valid iso field name. Allowed fields are ['p', 'rho', 'Mach', 'qcriterion', 's', 'T', 'Cp', 'mut', 'nuHat', 'Cpt', 'not_valid_field']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            _ = SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="po",
                        output_fields=["Cp"],
                        probe_points=[Point(name="pt1", location=(1, 2, 3))],
                    ),
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="iso1", field="Cpp", iso_value=0.5)],
                        output_fields=["primitiveVars"],
                    ),
                ],
                user_defined_fields=[
                    UserDefinedField(
                        name="not_valid_field", expression="primitiveVars[0] *3.1415926"
                    )
                ],
            )


def test_rotation_parent_volumes():

    c_1 = Cylinder(
        name="inner_rotating_cylinder",
        outer_radius=1 * u.cm,
        height=1 * u.cm,
        center=(0, 0, 0) * u.cm,
        axis=(0, 0, 1),
    )

    c_2 = Cylinder(
        name="outer_rotating_cylinder",
        outer_radius=12 * u.cm,
        height=12 * u.cm,
        center=(0, 0, 0) * u.cm,
        axis=(0, 0, 1),
    )

    c_3 = Cylinder(
        name="stationary_cylinder",
        outer_radius=12 * u.m,
        height=13 * u.m,
        center=(0, 0, 0) * u.m,
        axis=(0, 1, 2),
    )

    my_wall = Surface(name="my_wall", private_attribute_is_interface=False)

    msg = "For model #1, the parent rotating volume (stationary_cylinder) is not "
    "used in any other `Rotation` model's `volumes`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        with ValidationLevelContext(CASE):
            with SI_unit_system:
                SimulationParams(
                    models=[
                        Fluid(),
                        Rotation(entities=[c_1], spec=AngleExpression("1+2"), parent_volume=c_3),
                    ]
                )

    with ValidationLevelContext(CASE):
        with SI_unit_system:
            SimulationParams(
                models=[
                    Fluid(),
                    Rotation(entities=[c_1], spec=AngleExpression("1+2"), parent_volume=c_2),
                    Rotation(entities=[c_2], spec=AngleExpression("1+5")),
                    Wall(entities=[my_wall]),
                ],
                private_attribute_asset_cache=AssetCache(
                    project_length_unit="cm",
                    project_entity_info=VolumeMeshEntityInfo(boundaries=[my_wall]),
                ),
            )


def test_meshing_validator_dual_context():
    errors = None
    try:
        with SI_unit_system:
            with ValidationLevelContext(VOLUME_MESH):
                SimulationParams(meshing=None)
    except pd.ValidationError as err:
        errors = err.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["ctx"] == {"relevant_for": ["SurfaceMesh", "VolumeMesh"]}
    assert errors[0]["loc"] == ("meshing",)


def test_rotating_reference_frame_model_flag():

    c_1 = Cylinder(
        name="inner_rotating_cylinder",
        outer_radius=1 * u.cm,
        height=1 * u.cm,
        center=(0, 0, 0) * u.cm,
        axis=(0, 0, 1),
    )

    c_2 = Cylinder(
        name="outer_rotating_cylinder",
        outer_radius=12 * u.cm,
        height=12 * u.cm,
        center=(0, 0, 0) * u.cm,
        axis=(0, 0, 1),
    )

    c_3 = Cylinder(
        name="another_cylinder",
        outer_radius=12 * u.m,
        height=13 * u.m,
        center=(0, 0, 0) * u.m,
        axis=(0, 1, 2),
    )

    my_wall = Surface(name="my_wall", private_attribute_is_interface=False)
    timestepping_unsteady = Unsteady(steps=12, step_size=0.1 * u.s)
    timestepping_steady = Steady(max_steps=1000)

    msg = "For model #1, the rotating_reference_frame_model may not be set to False for "
    "steady state simulations."

    with pytest.raises(ValueError, match=re.escape(msg)):
        with ValidationLevelContext(CASE):
            with SI_unit_system:
                SimulationParams(
                    models=[
                        Fluid(),
                        Rotation(
                            entities=[c_1],
                            spec=AngleExpression("1+2"),
                            rotating_reference_frame_model=False,
                        ),
                        Wall(entities=[my_wall]),
                    ],
                    time_stepping=timestepping_steady,
                    private_attribute_asset_cache=AssetCache(
                        project_length_unit="cm",
                        project_entity_info=VolumeMeshEntityInfo(boundaries=[my_wall]),
                    ),
                )

    with ValidationLevelContext(CASE):
        with SI_unit_system:
            test_param = SimulationParams(
                models=[
                    Fluid(),
                    Rotation(
                        entities=[c_1],
                        spec=AngleExpression("1+2"),
                        parent_volume=c_2,
                        rotating_reference_frame_model=True,
                    ),
                    Rotation(
                        entities=[c_2],
                        spec=AngleExpression("1+5"),
                        rotating_reference_frame_model=False,
                    ),
                    Rotation(entities=[c_3], spec=AngleExpression("3+5")),
                    Wall(entities=[my_wall]),
                ],
                time_stepping=timestepping_unsteady,
                private_attribute_asset_cache=AssetCache(
                    project_length_unit="cm",
                    project_entity_info=VolumeMeshEntityInfo(boundaries=[my_wall]),
                ),
            )

    assert test_param.models[3].rotating_reference_frame_model == False


def test_output_fields_with_time_average_output():

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=[
                TimeAverageVolumeOutput(
                    name="TimeAverageVolume",
                    output_fields=["primitiveVars"],
                    start_step=4,
                    frequency=10,
                    frequency_offset=14,
                ),
                TimeAverageSurfaceOutput(
                    name="TimeAverageSurface",
                    output_fields=["primitiveVars"],
                    entities=[
                        Surface(name="VOLUME/LEFT"),
                    ],
                    start_step=4,
                    frequency=10,
                    frequency_offset=14,
                ),
                TimeAverageSurfaceOutput(
                    name="TimeAverageSurface",
                    output_fields=["T"],
                    entities=[
                        Surface(name="VOLUME/RIGHT"),
                    ],
                    start_step=4,
                    frequency=10,
                    frequency_offset=14,
                ),
                TimeAverageSliceOutput(
                    entities=[
                        Slice(
                            name="TimeAverageSlice",
                            origin=(0, 0, 0) * u.m,
                            normal=(0, 0, 1),
                        )
                    ],
                    output_fields=["s", "T"],
                    start_step=4,
                    frequency=10,
                    frequency_offset=14,
                ),
            ],
        )

    assert params

    # Invalid simulation params
    output_type_set = set()
    for output in params.outputs:
        output_type_set.add(f"`{output.output_type}`")
    output_type_list = ",".join(sorted(output_type_set)).strip(",")
    message = f"{output_type_list} can only be used in unsteady simulations."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        params.time_stepping = Steady(max_steps=1000)


def test_wall_deserialization():
    # Wall->velocity accept discriminated AND non-discriminated unions.
    # Need to check if all works when deserializing.
    dummy_boundary = Surface(name="chameleon")
    simple_wall = Wall(**Wall(entities=dummy_boundary).model_dump(mode="json"))
    assert simple_wall.velocity is None

    const_vel_wall = Wall(
        **Wall(entities=dummy_boundary, velocity=[1, 2, 3] * u.m / u.s).model_dump(mode="json")
    )
    assert all(const_vel_wall.velocity == [1, 2, 3] * u.m / u.s)

    slater_bleed_wall = Wall(
        **Wall(
            entities=dummy_boundary,
            velocity=SlaterPorousBleed(porosity=0.2, static_pressure=0.1 * u.Pa),
        ).model_dump(mode="json")
    )
    assert slater_bleed_wall.velocity.porosity == 0.2
    assert slater_bleed_wall.velocity.static_pressure == 0.1 * u.Pa
