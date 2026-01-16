import json
import re
import unittest

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemStatus,
)
from flow360.component.simulation.draft_context.mirror import MirrorPlane, MirrorStatus
from flow360.component.simulation.entity_info import (
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.entity_selector import (
    SurfaceSelector,
    collect_and_tokenize_selectors_in_place,
)
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.edge_params import (
    HeightBasedRefinement,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    GeometryRefinement,
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    MeshingDefaults,
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
    VolumeMeshingParams,
    snappy,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    UserDefinedFarfield,
)
from flow360.component.simulation.models.material import SolidMaterial, aluminum
from flow360.component.simulation.models.solver_numerics import (
    DetachedEddySimulation,
    KOmegaSST,
    KOmegaSSTModelConstants,
    SpalartAllmaras,
    SpalartAllmarasModelConstants,
    TransitionModelSolver,
    TurbulenceModelControls,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    HeatFlux,
    Inflow,
    Outflow,
    Periodic,
    PorousJump,
    Pressure,
    SlaterPorousBleed,
    SlipWall,
    TotalPressure,
    Translational,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    AngularVelocity,
    Fluid,
    HeatEquationInitialCondition,
    NavierStokesInitialCondition,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    LiquidOperatingCondition,
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
    TimeAverageIsosurfaceOutput,
    TimeAverageSliceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    Box,
    CustomVolume,
    Cylinder,
    Edge,
    GenericVolume,
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    SeedpointVolume,
    Surface,
    SurfacePrivateAttributes,
)
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.component.simulation.validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
    ParamsValidationInfo,
    ValidationContext,
)

quasi_3d_farfield_context = ParamsValidationInfo({}, [])
quasi_3d_farfield_context.farfield_method = "quasi-3d"
quasi_3d_periodic_farfield_context = ParamsValidationInfo({}, [])
quasi_3d_periodic_farfield_context.farfield_method = "quasi-3d-periodic"

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def reset_context():
    clear_context()


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


def test_hybrid_model_to_use_zonal_enforcement(fluid_model, fluid_model_with_hybrid_model):

    fluid_model_with_hybrid_model.turbulence_model_solver.controls = [
        TurbulenceModelControls(enforcement="RANS", entities=[GenericVolume(name="block-1")])
    ]

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_hybrid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )

    assert params

    fluid_model.turbulence_model_solver.controls = [
        TurbulenceModelControls(enforcement="RANS", entities=[GenericVolume(name="block-1")])
    ]

    message = "Control region 0 must be running in hybrid RANS-LES mode to apply zonal turbulence enforcement."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            models=[fluid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )


def test_zonal_modeling_constants_consistency(fluid_model_with_hybrid_model):
    fluid_model_with_hybrid_model.turbulence_model_solver.controls = [
        TurbulenceModelControls(
            enforcement="RANS",
            modeling_constants=SpalartAllmarasModelConstants(),
            entities=[GenericVolume(name="block-1")],
        )
    ]

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_hybrid_model],
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
        )

    assert params

    message = "Turbulence model is SpalartAllmaras, but controls.modeling_constants is of a "
    "conflicting class in control region 0."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        TurbulenceModelSolver = SpalartAllmaras(
            controls=[
                TurbulenceModelControls(
                    enforcement="LES",
                    modeling_constants=KOmegaSSTModelConstants(),
                    entities=[GenericVolume(name="block-1")],
                )
            ]
        )

    message = "Turbulence model is KOmegaSST, but controls.modeling_constants is of a "
    "conflicting class in control region 0."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        TurbulenceModelSolver = KOmegaSST(
            controls=[
                TurbulenceModelControls(
                    enforcement="LES",
                    modeling_constants=SpalartAllmarasModelConstants(),
                    entities=[GenericVolume(name="block-1")],
                )
            ]
        )


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


def test_BC_geometry():
    """For a quasi 3D geometry test the check for the"""
    # --------------------------------------------------------#
    # >>>>>>> Group sides of airfoil as individual ones <<<<<<<
    with open("./data/geometry_metadata_asset_cache_quasi3D.json") as fp:
        data = json.load(fp)
        data["private_attribute_asset_cache"]["project_entity_info"]["face_group_tag"] = "groupName"
        # Mock private_attributes for all boundaries
        for group in data["private_attribute_asset_cache"]["project_entity_info"]["grouped_faces"]:
            for face in group:
                if "private_attributes" not in face:
                    face["private_attributes"] = {"bounding_box": [[0, 0, 0], [1, 1, 1]]}
        asset_cache = AssetCache(**data["private_attribute_asset_cache"])

    symmetry_boundary_1 = [item for item in asset_cache.boundaries if item.name == "symmetry11"][0]
    symmetry_boundary_2 = [item for item in asset_cache.boundaries if item.name == "symmetry22"][0]
    wall = [item for item in asset_cache.boundaries if item.name == "wall"][0]

    ##### AUTO METHOD #####
    auto_farfield = AutomatedFarfield(name="my_farfield")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                Wall(entities=wall),
                SlipWall(entities=auto_farfield.symmetry_planes),
                SlipWall(entities=symmetry_boundary_1),
                SlipWall(entities=symmetry_boundary_2),
                Freestream(entities=auto_farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["loc"] == ("models", 3, "entities")
    assert errors[0]["msg"] == (
        "Value error, Boundary `symmetry11` will likely be deleted after mesh generation. "
        "Therefore it cannot be used."
    )

    ##### Ghost entities not used #####
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                Wall(entities=wall),
                SlipWall(entities=symmetry_boundary_2),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )

    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: farfield, symmetric."
        " Please add them to a boundary condition model in the `models` section."
    )

    ##### QUASI 3D METHOD #####
    auto_farfield = AutomatedFarfield(name="my_farfield", method="quasi-3d")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                Wall(entities=wall),
                SlipWall(entities=auto_farfield.symmetry_planes),
                SlipWall(entities=symmetry_boundary_1),
                SlipWall(entities=symmetry_boundary_2),
                Freestream(entities=auto_farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 2
    assert errors[0]["loc"] == ("models", 3, "entities")
    assert errors[0]["msg"] == (
        "Value error, Boundary `symmetry11` will likely be deleted after mesh generation. "
        "Therefore it cannot be used."
    )
    assert errors[1]["loc"] == ("models", 4, "entities")
    assert errors[1]["msg"] == (
        "Value error, Boundary `symmetry22` will likely be deleted after mesh generation. "
        "Therefore it cannot be used."
    )
    ##### Ghost entities not used #####
    auto_farfield = AutomatedFarfield(name="my_farfield", method="quasi-3d")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                Wall(entities=wall),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: farfield, "
        "symmetric-1, symmetric-2. Please add them to a boundary condition model in the `models` section."
    )

    # --------------------------------------------------------#
    # >>>>>>> Group sides of airfoil as SINGLE boundary <<<<<<<
    # This is a known defect of the current deletion detection logic.
    # Documentation is updated to reflect this.


def test_incomplete_BC_volume_mesh():
    ##:: Construct a dummy asset cache
    wall_1 = Surface(
        name="wall_1", private_attribute_is_interface=False, private_attribute_id="wall_1"
    )
    periodic_1 = Surface(
        name="periodic_1", private_attribute_is_interface=False, private_attribute_id="periodic_1"
    )
    periodic_2 = Surface(
        name="periodic_2", private_attribute_is_interface=False, private_attribute_id="periodic_2"
    )
    i_exist = Surface(
        name="i_exist", private_attribute_is_interface=False, private_attribute_id="i_exist"
    )
    no_bc = Surface(
        name="no_bc", private_attribute_is_interface=False, private_attribute_id="no_bc"
    )
    some_interface = Surface(
        name="some_interface",
        private_attribute_is_interface=True,
        private_attribute_id="some_interface",
    )

    asset_cache = AssetCache(
        project_length_unit="inch",
        project_entity_info=VolumeMeshEntityInfo(
            boundaries=[wall_1, periodic_1, periodic_2, i_exist, some_interface, no_bc]
        ),
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                )
            ),
            models=[
                Fluid(),
                # Stage 1.5: Use selector instead of explicit entity to test BC validation
                Wall(entities=[SurfaceSelector(name="wall_selector").match("wall_*")]),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[i_exist]),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    submission_ready_dict = collect_and_tokenize_selectors_in_place(
        params.model_dump(mode="json", exclude_none=True)
    )
    params, errors, _ = validate_model(
        params_as_dict=submission_ready_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: no_bc. "
        "Please add them to a boundary condition model in the `models` section."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                )
            ),
            models=[
                Fluid(),
                # Stage 1.5: Mix selector with explicit entity
                Wall(entities=[SurfaceSelector(name="wall_selector").match("wall_*"), i_exist]),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[Surface(name="plz_dont_do_this"), no_bc]),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    submission_ready_dict = collect_and_tokenize_selectors_in_place(
        params.model_dump(mode="json", exclude_none=True)
    )
    params, errors, _ = validate_model(
        params_as_dict=submission_ready_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries are not known `Surface` entities "
        "but appear in the `models` section: plz_dont_do_this."
    )


def test_incomplete_BC_surface_mesh():
    ##:: Construct a dummy asset cache
    dummy_attrs = SurfacePrivateAttributes(bounding_box=[[0, 0, 0], [1, 1, 1]])
    wall_1 = Surface(
        name="wall_1", private_attribute_is_interface=False, private_attributes=dummy_attrs
    )
    periodic_1 = Surface(
        name="periodic_1", private_attribute_is_interface=False, private_attributes=dummy_attrs
    )
    periodic_2 = Surface(
        name="periodic_2", private_attribute_is_interface=False, private_attributes=dummy_attrs
    )
    i_exist = Surface(
        name="i_exist", private_attribute_is_interface=False, private_attributes=dummy_attrs
    )
    no_bc = Surface(
        name="no_bc", private_attribute_is_interface=False, private_attributes=dummy_attrs
    )
    i_will_be_deleted = Surface(
        name="sym_boundary",
        private_attribute_is_interface=False,
        private_attributes=SurfacePrivateAttributes(
            bounding_box=[[0, -1e-6, 0], [1, 1e-8, 1]],
        ),
    )
    auto_farfield = AutomatedFarfield(name="my_farfield")

    asset_cache = AssetCache(
        project_length_unit="inch",
        project_entity_info=SurfaceMeshEntityInfo(
            boundaries=[wall_1, periodic_1, periodic_2, i_exist, no_bc, i_will_be_deleted],
            ghost_entities=[
                GhostSphere(name="farfield"),
                GhostCircularPlane(name="symmetric", center=[-1, 0, 0], maxRadius=100),
                GhostCircularPlane(name="symmetric-1", center=[-1, -100, 0], maxRadius=100),
                GhostCircularPlane(name="symmetric-2", center=[-1, 1e-12, 0], maxRadius=100),
            ],
            global_bounding_box=[[-100, -100, -100], [100, 1e-12, 100]],
        ),
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                # Stage 1.5: Use selector instead of explicit entity
                Wall(entities=[SurfaceSelector(name="wall_selector").match("wall_*")]),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[i_exist]),
                SlipWall(entities=[no_bc]),
                Freestream(entities=auto_farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    # i_will_be_deleted won't trigger "no bc assigned" error but `symmetric` will.
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1, print(">>>", errors)
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: symmetric. "
        "Please add them to a boundary condition model in the `models` section."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                # Stage 1.5: Use selector for wall
                Wall(entities=[SurfaceSelector(name="wall_selector").match("wall_*")]),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[auto_farfield.symmetry_planes]),
                SlipWall(entities=[i_exist]),
                Freestream(entities=auto_farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1, print(">>>", errors)
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: no_bc. "
        "Please add them to a boundary condition model in the `models` section."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            models=[
                Fluid(),
                Wall(entities=[wall_1]),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[i_exist]),
                SlipWall(entities=[auto_farfield.symmetry_planes]),
                Freestream(entities=auto_farfield.farfield),
                SlipWall(entities=[Surface(name="plz_dont_do_this"), no_bc]),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1, print(">>>", errors)
    assert errors[0]["msg"] == (
        "Value error, The following boundaries are not known `Surface` entities "
        "but appear in the `models` section: plz_dont_do_this."
    )


def test_porousJump_entities_is_interface(mock_validation_context):
    surface_1_is_interface = Surface(name="Surface-1", private_attribute_is_interface=True)
    surface_2_is_not_interface = Surface(name="Surface-2", private_attribute_is_interface=False)
    surface_3_is_interface = Surface(name="Surface-3", private_attribute_is_interface=True)
    error_message = "Boundary `Surface-2` is not an interface"
    with mock_validation_context, pytest.raises(ValueError, match=re.escape(error_message)):
        PorousJump(
            entity_pairs=[(surface_1_is_interface, surface_2_is_not_interface)],
            darcy_coefficient=1e6 / (u.m * u.m),
            forchheimer_coefficient=1e3 / u.m,
            thickness=0.01 * u.m,
        )

    with mock_validation_context, pytest.raises(ValueError, match=re.escape(error_message)):
        PorousJump(
            entity_pairs=[(surface_2_is_not_interface, surface_1_is_interface)],
            darcy_coefficient=1e6,
            forchheimer_coefficient=1e3,
            thickness=0.01,
        )

    PorousJump(
        entity_pairs=[(surface_1_is_interface, surface_3_is_interface)],
        darcy_coefficient=1e6 / (u.m * u.m),
        forchheimer_coefficient=1e3 / u.m,
        thickness=0.01 * u.m,
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
        volumes=[entity_generic_volume, entity_generic_volume],
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
        spec=AngleExpression("sin(2*t)"),
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

    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    # Invalid simulation params
    with SI_unit_system, mock_context, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[volume_model1, volume_model2, surface_model1, surface_model2, surface_model3],
        )

    message = f"Volume entity `{entity_cylinder.name}` appears multiple times in `{rotation_model1.type}` model.\n"

    # Invalid simulation params (Draft Entity)
    with SI_unit_system, mock_context, pytest.raises(ValueError, match=re.escape(message)):
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
    msg = "In `outputs`[0] SurfaceOutput:, not_valid_field is not a valid output field name. Allowed fields are "
    "['Cp', "
    "'Cpt', "
    "'gradW', "
    "'kOmega', "
    "'Mach', "
    "'mut', "
    "'mutRatio', "
    "'nuHat', "
    "'primitiveVars', "
    "'qcriterion', "
    "'residualNavierStokes', "
    "'residualTransition', "
    "'residualTurbulence', "
    "'s', "
    "'solutionNavierStokes', "
    "'solutionTransition', "
    "'solutionTurbulence', "
    "'T', "
    "'velocity', "
    "'velocity_x', "
    "'velocity_y', "
    "'velocity_z', "
    "'velocity_magnitude', "
    "'pressure', "
    "'vorticity', "
    "'vorticityMagnitude', "
    "'wallDistance', "
    "'numericalDissipationFactor', "
    "'residualHeatSolver', "
    "'VelocityRelative', "
    "'lowMachPreconditionerSensor', "
    "'velocity_m_per_s', "
    "'velocity_x_m_per_s', "
    "'velocity_y_m_per_s', "
    "'velocity_z_m_per_s', "
    "'velocity_magnitude_m_per_s', "
    "'pressure_pa', "
    "'CfVec', "
    "'Cf', "
    "'heatFlux', "
    "'nodeNormals', "
    "'nodeForcesPerUnitArea', "
    "'yPlus', "
    "'wallFunctionMetric', "
    "'heatTransferCoefficientStaticTemperature', "
    "'heatTransferCoefficientTotalTemperature', "
    "'wall_shear_stress_magnitude', "
    "'wall_shear_stress_magnitude_pa']."
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

    msg = "In `outputs`[1] SliceOutput:, not_valid_field_2 is not a valid output field name. Allowed fields are "
    "['Cp', "
    "'Cpt', "
    "'gradW', "
    "'kOmega', "
    "'Mach', "
    "'mut', "
    "'mutRatio', "
    "'nuHat', "
    "'primitiveVars', "
    "'qcriterion', "
    "'residualNavierStokes', "
    "'residualTransition', "
    "'residualTurbulence', "
    "'s', "
    "'solutionNavierStokes', "
    "'solutionTransition', "
    "'solutionTurbulence', "
    "'T', "
    "'velocity', "
    "'velocity_x', "
    "'velocity_y', "
    "'velocity_z', "
    "'velocity_magnitude', "
    "'pressure', "
    "'vorticity', "
    "'vorticityMagnitude', "
    "'wallDistance', "
    "'numericalDissipationFactor', "
    "'residualHeatSolver', "
    "'VelocityRelative', "
    "'lowMachPreconditionerSensor', "
    "'velocity_m_per_s', "
    "'velocity_x_m_per_s', "
    "'velocity_y_m_per_s', "
    "'velocity_z_m_per_s', "
    "'velocity_magnitude_m_per_s', "
    "'pressure_pa', "
    "'betMetrics', "
    "'betMetricsPerDisk', "
    "'linearResidualNavierStokes', "
    "'linearResidualTurbulence', "
    "'linearResidualTransition', "
    "'SpalartAllmaras_hybridModel', "
    "'kOmegaSST_hybridModel', "
    "'localCFL', "
    "'not_valid_field']."
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

    msg = "The legacy string output fields in `SurfaceIntegralOutput` must be used with `UserDefinedField`."
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

    with SI_unit_system:
        params = SimulationParams(
            outputs=[
                SurfaceIntegralOutput(
                    name="MassFluxIntegral",
                    output_fields=[
                        UserVariable(
                            name="MassFluxProjected",
                            value=-1
                            * solution.density
                            * math.dot(solution.velocity, solution.node_area_vector),
                        )
                    ],
                    surfaces=[surface_1],
                )
            ]
        )

    msg = (
        "In `outputs`[1] IsosurfaceOutput:, Cpp is not a valid iso field name. Allowed fields are "
    )
    "['p', "
    "'rho', "
    "'Mach', "
    "'qcriterion', "
    "'s', "
    "'T', "
    "'Cp', "
    "'Cpt', "
    "'mut', "
    "'nuHat', "
    "'vorticityMagnitude', "
    "'not_valid_field']."
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


def test_rotation_parent_volumes(mock_case_validation_context):

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
    with mock_case_validation_context, pytest.raises(ValueError, match=re.escape(msg)):
        with SI_unit_system:
            SimulationParams(
                models=[
                    Fluid(),
                    Rotation(entities=[c_1], spec=AngleExpression("1+2"), parent_volume=c_3),
                ]
            )

    with ValidationContext(CASE):
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
            with ValidationContext(VOLUME_MESH):
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
        with ValidationContext(CASE):
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

    with ValidationContext(CASE):
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


def test_populate_validated_models_to_validation_context(mock_validation_context):
    """Test that models are properly populated to validation context."""
    # Create models with private_attribute_id
    fluid_model = Fluid()
    wall_model = Wall(
        name="wall_bc",
        surfaces=[Surface(name="wall_surface")],
    )

    # Before validation, physics_model_dict should be None
    assert mock_validation_context.info.physics_model_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(
            models=[fluid_model, wall_model],
        )

    # After validation, physics_model_dict should be populated
    assert mock_validation_context.info.physics_model_dict is not None
    assert isinstance(mock_validation_context.info.physics_model_dict, dict)

    # Check that models are in the dict with their IDs as keys
    assert len(mock_validation_context.info.physics_model_dict) == 2
    assert fluid_model.private_attribute_id in mock_validation_context.info.physics_model_dict
    assert wall_model.private_attribute_id in mock_validation_context.info.physics_model_dict

    # Verify the objects are the same
    assert (
        mock_validation_context.info.physics_model_dict[fluid_model.private_attribute_id]
        == fluid_model
    )
    assert (
        mock_validation_context.info.physics_model_dict[wall_model.private_attribute_id]
        == wall_model
    )


def test_populate_validated_outputs_to_validation_context(mock_validation_context):
    """Test that outputs are properly populated to validation context."""
    # Create outputs with private_attribute_id
    probe_output = ProbeOutput(
        name="probe1",
        output_fields=["Cp"],
        probe_points=[Point(name="pt1", location=(1, 2, 3) * u.m)],
    )

    surface_output = SurfaceOutput(
        name="surface1",
        output_fields=["Cp"],
        entities=[Surface(name="wall")],
    )

    volume_output = VolumeOutput(
        name="volume1",
        output_fields=["primitiveVars"],
    )

    # Before validation, output_dict should be None
    assert mock_validation_context.info.output_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(
            outputs=[probe_output, surface_output, volume_output],
        )

    # After validation, output_dict should be populated
    assert mock_validation_context.info.output_dict is not None
    assert isinstance(mock_validation_context.info.output_dict, dict)

    # Check that outputs are in the dict with their IDs as keys
    assert len(mock_validation_context.info.output_dict) == 3
    assert probe_output.private_attribute_id in mock_validation_context.info.output_dict
    assert surface_output.private_attribute_id in mock_validation_context.info.output_dict
    assert volume_output.private_attribute_id in mock_validation_context.info.output_dict

    # Verify the objects are the same
    assert (
        mock_validation_context.info.output_dict[probe_output.private_attribute_id] == probe_output
    )
    assert (
        mock_validation_context.info.output_dict[surface_output.private_attribute_id]
        == surface_output
    )
    assert (
        mock_validation_context.info.output_dict[volume_output.private_attribute_id]
        == volume_output
    )


def test_populate_both_models_and_outputs_to_validation_context(mock_validation_context):
    """Test that both models and outputs are properly populated to the same validation context."""
    # Create models and outputs
    fluid_model = Fluid()
    probe_output = ProbeOutput(
        name="probe1",
        output_fields=["Cp"],
        probe_points=[Point(name="pt1", location=(1, 2, 3) * u.m)],
    )

    # Before validation, both should be None
    assert mock_validation_context.info.physics_model_dict is None
    assert mock_validation_context.info.output_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(
            models=[fluid_model],
            outputs=[probe_output],
        )

    # After validation, both should be populated
    assert mock_validation_context.info.physics_model_dict is not None
    assert mock_validation_context.info.output_dict is not None

    # Verify both dicts are populated correctly
    assert fluid_model.private_attribute_id in mock_validation_context.info.physics_model_dict
    assert probe_output.private_attribute_id in mock_validation_context.info.output_dict

    assert (
        mock_validation_context.info.physics_model_dict[fluid_model.private_attribute_id]
        == fluid_model
    )
    assert (
        mock_validation_context.info.output_dict[probe_output.private_attribute_id] == probe_output
    )


def test_populate_outputs_none_sets_empty_dict(mock_validation_context):
    """Test that output_dict is set to {} when outputs=None.

    This distinguishes successful validation with no outputs (output_dict={})
    from validation errors (output_dict=None).
    """
    assert mock_validation_context.info.output_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(outputs=None)

    # output_dict should be set to empty dict, not None
    assert mock_validation_context.info.output_dict == {}


def test_populate_outputs_empty_list_sets_empty_dict(mock_validation_context):
    """Test that output_dict is set to {} when outputs=[]."""
    assert mock_validation_context.info.output_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(outputs=[])

    # output_dict should be set to empty dict
    assert mock_validation_context.info.output_dict == {}


def test_populate_models_none_sets_dict_with_default(mock_validation_context):
    """Test that physics_model_dict is populated when models=None.

    Note: SimulationParams automatically adds a default Fluid model when models=None,
    so physics_model_dict will contain the default model, not be empty.
    This still distinguishes successful validation from validation errors (physics_model_dict=None).
    """
    assert mock_validation_context.info.physics_model_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(models=None)

    # physics_model_dict should be populated with default Fluid model
    assert mock_validation_context.info.physics_model_dict is not None
    assert isinstance(mock_validation_context.info.physics_model_dict, dict)
    # Should contain the default fluid model
    assert len(mock_validation_context.info.physics_model_dict) == 1
    assert "__default_fluid" in mock_validation_context.info.physics_model_dict


def test_populate_models_empty_list_sets_dict_with_default(mock_validation_context):
    """Test that physics_model_dict is populated when models=[].

    Note: SimulationParams automatically adds a default Fluid model when models=[],
    so physics_model_dict will contain the default model.
    """
    assert mock_validation_context.info.physics_model_dict is None

    with SI_unit_system, mock_validation_context:
        params = SimulationParams(models=[])

    # physics_model_dict should be populated with default Fluid model
    assert mock_validation_context.info.physics_model_dict is not None
    assert isinstance(mock_validation_context.info.physics_model_dict, dict)
    assert len(mock_validation_context.info.physics_model_dict) == 1
    assert "__default_fluid" in mock_validation_context.info.physics_model_dict


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_deleted_surfaces():
    with open("./data/geometry_metadata_asset_cache.json") as fp:
        asset_cache = AssetCache(**json.load(fp))

    all_boundaries: list[Surface] = asset_cache.project_entity_info.get_boundaries()

    # OverlapQuasi2DSymmetric & OverlapHalfModelSymmetric
    overlap_with_two_symmetric = all_boundaries[2]
    # OverlapHalfModelSymmetric
    overlap_with_single_symmetric = all_boundaries[3]

    with SI_unit_system:
        farfield = AutomatedFarfield(method="auto")
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4, surface_max_edge_length=1e-2
                ),
                volume_zones=[farfield],
            ),
            operating_condition=AerospaceCondition(velocity_magnitude=0.2),
            models=[
                Wall(entities=all_boundaries[0]),
                Wall(entities=all_boundaries[1]),
                SlipWall(entities=overlap_with_two_symmetric),
                SlipWall(entities=[overlap_with_single_symmetric, farfield.symmetry_planes]),
                Freestream(entities=farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )

    assert len(errors) == 1
    assert (
        errors[0]["msg"] == "Value error, Boundary `body0001_face0003` will likely"
        " be deleted after mesh generation. Therefore it cannot be used."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4, surface_max_edge_length=1e-2
                ),
                volume_zones=[AutomatedFarfield(method="quasi-3d")],
            ),
            operating_condition=AerospaceCondition(velocity_magnitude=0.2),
            models=[
                Wall(entities=all_boundaries[0]),
                Wall(entities=all_boundaries[1]),
                Periodic(
                    surface_pairs=[(overlap_with_single_symmetric, overlap_with_two_symmetric)],
                    spec=Translational(),
                ),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, Boundaries `body0001_face0004`, `body0001_face0003` will likely "
        + "be deleted after mesh generation. Therefore they cannot be used."
    )
    assert errors[0]["loc"] == ("models", 2, "entity_pairs")


def test_validate_liquid_operating_condition():
    with open("./data/geometry_metadata_asset_cache.json") as fp:
        asset_cache = AssetCache(**json.load(fp))
    all_boundaries: list[Surface] = asset_cache.project_entity_info.get_boundaries()
    with u.SI_unit_system:
        porous_zone = GenericVolume(name="zone_with_no_axes")
        porous_zone.axes = [[0, 1, 0], [0, 0, 1]]
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Fluid(
                    initial_condition=NavierStokesInitialCondition(
                        rho="1;",
                    )
                ),
                PorousMedium(
                    volumes=[porous_zone],
                    darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
                    forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
                    volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
                ),
                Wall(
                    heat_spec=HeatFlux(value=10 * u.W / u.m**2),
                    surfaces=all_boundaries[0:-1],
                    velocity=["1", "2", "2"],
                ),
                Outflow(entities=all_boundaries[-1], spec=Pressure(value=1.01e6 * u.Pa)),
                Rotation(
                    volumes=[
                        Cylinder(
                            name="Cylinder",
                            outer_radius=1 * u.cm,
                            height=1 * u.cm,
                            center=(0, 0, 0) * u.cm,
                            axis=(0, 0, 1),
                            private_attribute_id="1",
                        )
                    ],
                    name="rotation",
                    spec=AngleExpression("sin(t)"),
                ),
            ],
            outputs=[VolumeOutput(output_fields=["T"])],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )
    assert len(errors) == 5
    assert (
        errors[0]["msg"]
        == "Value error, Expression cannot be used when using liquid as simulation material."
    )
    assert errors[0]["loc"] == ("models", 0, "initial_condition", "rho")
    assert (
        errors[1]["msg"]
        == "Value error, `volumetric_heat_source` cannot be setup under `PorousMedium` when using liquid as simulation material."
    )
    assert errors[1]["loc"] == ("models", 1, "volumetric_heat_source")

    assert (
        errors[2]["msg"]
        == "Value error, Expression cannot be used when using liquid as simulation material."
    )
    assert errors[2]["loc"] == ("models", 2, "velocity")
    assert (
        errors[3]["msg"]
        == "Value error, Only adiabatic wall is allowed when using liquid as simulation material."
    )
    assert errors[3]["loc"] == ("models", 2, "heat_spec")
    assert (
        errors[4]["msg"]
        == "Value error, Output field T cannot be selected when using liquid as simulation material."
    )
    assert errors[4]["loc"] == ("outputs", 0, "output_fields")

    with u.SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Outflow(entities=all_boundaries[-1], spec=Pressure(value=1.01e6 * u.Pa)),
            ],
            user_defined_dynamics=[
                UserDefinedDynamic(
                    name="alphaController",
                    input_vars=["CL"],
                    constants={"CLTarget": 0.4, "Kp": 0.2, "Ki": 0.002},
                    output_vars={"alphaAngle": "if (pseudoStep > 500) state[0]; else alphaAngle;"},
                    state_vars_initial_value=["alphaAngle", "0.0"],
                    update_law=[
                        "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
                        "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
                    ],
                    input_boundary_patches=[Surface(name="UDDPatch")],
                )
            ],
            outputs=[
                VolumeOutput(
                    output_format="both",
                    output_fields=["four"],
                ),
            ],
            user_defined_fields=[
                UserDefinedField(
                    name="four",
                    expression="2+2",
                ),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    assert len(errors) == 2
    assert (
        errors[0]["msg"]
        == "Value error, `Outflow` type model cannot be used when using liquid as simulation material."
    )
    assert errors[0]["loc"] == ("models",)
    assert (
        errors[1]["msg"]
        == "Value error, user_defined_fields cannot be used when using liquid as simulation material."
    )
    assert errors[1]["loc"] == ("user_defined_fields",)

    with u.SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Inflow(
                    entities=[all_boundaries[-1]],
                    total_temperature=300 * u.K,
                    spec=TotalPressure(
                        value=1.028e6 * u.Pa,
                    ),
                    velocity_direction=(1, 0, 0),
                )
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, `Inflow` type model cannot be used when using liquid as simulation material."
    )
    assert errors[0]["loc"] == ("models",)

    with u.SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Solid(
                    volumes=[GenericVolume(name="CHTSolid")],
                    material=aluminum,
                    volumetric_heat_source="0",
                    initial_condition=HeatEquationInitialCondition(temperature="10"),
                ),
            ],
            private_attribute_asset_cache=asset_cache,
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, `Solid` type model cannot be used when using liquid as simulation material."
    )
    assert errors[0]["loc"] == ("models",)


def test_beta_mesher_only_features(mock_validation_context):
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4, surface_max_edge_length=1e-2
                ),
                refinements=[
                    BoundaryLayer(
                        faces=[Surface(name="face1"), Surface(name="face2")],
                        growth_rate=1.1,
                    )
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=False),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 2
    assert errors[0]["msg"] == ("Value error, First layer thickness is required.")
    assert errors[1]["msg"] == (
        "Value error, Growth rate per face is only supported by the beta mesher."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    number_of_boundary_layers=10,
                ),
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=False),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, Number of boundary layers is only supported by the beta mesher."
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    planar_face_tolerance=1e-4,
                ),
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=False),
        )
    params, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert errors is None

    # Using CustomZones without UserDefinedFarfield
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
                            CustomVolume(
                                name="zone1",
                                boundaries=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                    AutomatedFarfield(),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, CustomVolume is only supported when "
        + "beta mesher and user defined farfield are enabled."
    )

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
                            CustomVolume(
                                name="zone1",
                                boundaries=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                    UserDefinedFarfield(),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=False),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, CustomVolume is only supported when "
        + "beta mesher and user defined farfield are enabled."
    )

    # Unique volume zone names
    beta_mesher_context = ParamsValidationInfo({}, [])
    beta_mesher_context.is_beta_mesher = True
    beta_mesher_context.farfield_method = "user-defined"
    # Needed for per-entity validation of CustomVolume/SeedpointVolume when instantiated under a
    # manually constructed ParamsValidationInfo (i.e., outside validate_model()).
    beta_mesher_context.to_be_generated_custom_volumes = {"zone1"}

    with ValidationContext(SURFACE_MESH, beta_mesher_context), pytest.raises(
        ValueError, match="Multiple CustomVolume with the same name `zone1` are not allowed."
    ):
        with SI_unit_system:
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=1e-4,
                        planar_face_tolerance=1e-4,
                        surface_max_edge_length=1e-5,
                    ),
                    volume_zones=[
                        CustomZones(
                            name="custom_zones",
                            entities=[
                                CustomVolume(
                                    name="zone1",
                                    boundaries=[Surface(name="face1"), Surface(name="face2")],
                                ),
                                CustomVolume(
                                    name="zone1",
                                    boundaries=[Surface(name="face3"), Surface(name="face4")],
                                ),
                            ],
                        ),
                        UserDefinedFarfield(),
                    ],
                ),
                private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
            )

    # Unique interface names
    with mock_validation_context, pytest.raises(
        ValueError, match="The boundaries of a CustomVolume must have different names."
    ):
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
                                CustomVolume(
                                    name="zone1",
                                    boundaries=[Surface(name="face1"), Surface(name="face1")],
                                )
                            ],
                        ),
                        UserDefinedFarfield(),
                    ],
                ),
                private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
            )

    # Ensure that the boundaries of CustomVolume do not require a boundary condition
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
                            CustomVolume(
                                name="zone1",
                                boundaries=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                    UserDefinedFarfield(),
                ],
            ),
            models=[
                Wall(
                    entities=[
                        Surface(name="face2"),
                    ]
                ),
            ],
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=True,
                project_entity_info=SurfaceMeshEntityInfo(
                    boundaries=[
                        Surface(
                            name="face1",
                            private_attributes=SurfacePrivateAttributes(
                                bounding_box=[[0, 0, 0], [1, 1, 1]]
                            ),
                        ),
                        Surface(
                            name="face2",
                            private_attributes=SurfacePrivateAttributes(
                                bounding_box=[[0, 0, 0], [1, 1, 1]]
                            ),
                        ),
                        Surface(
                            name="face3",
                            private_attributes=SurfacePrivateAttributes(
                                bounding_box=[[0, 0, 0], [1, 1, 1]]
                            ),
                        ),
                    ]
                ),
            ),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: face3. "  # Face1 should not be here
        "Please add them to a boundary condition model in the `models` section."
    )
    assert errors[0]["loc"] == ()


def test_geometry_AI_only_features():
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    geometry_accuracy=1e-5 * u.m,
                    surface_max_aspect_ratio=20.0,
                    surface_max_adaptation_iterations=20,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=1e-5 * u.m, entities=[Surface(name="noSlipWall")]
                    )
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=False, use_geometry_AI=False
            ),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 4
    assert (
        errors[0]["msg"]
        == "Value error, Geometry accuracy is only supported when geometry AI is used."
    )
    assert (
        errors[1]["msg"]
        == "Value error, surface_max_aspect_ratio is only supported when geometry AI is used."
    )
    assert (
        errors[2]["msg"]
        == "Value error, surface_max_adaptation_iterations is only supported when geometry AI is used."
    )
    assert errors[3]["msg"] == "Value error, GeometryRefinement is only supported by geometry AI."

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=False, use_geometry_AI=True
            ),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"] == "Value error, Geometry accuracy is required when geometry AI is used."
    )


def test_geometry_AI_unsupported_features():
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    geometry_accuracy=1e-4 * u.m,
                    surface_max_aspect_ratio=20.0,
                    surface_max_adaptation_iterations=20,
                ),
                refinements=[
                    SurfaceEdgeRefinement(
                        edges=[Edge(name="edge0001")], method=HeightBasedRefinement(value=1e-4)
                    )
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=False, use_geometry_AI=True
            ),
        )
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"]
        == "Value error, SurfaceEdgeRefinement is not currently supported with geometry AI."
    )


def test_redefined_user_defined_fields():

    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(
                velocity_magnitude=100.0 * u.m / u.s,
            ),
            outputs=[
                VolumeOutput(
                    output_format="both",
                    output_fields=["pressure"],
                ),
            ],
            user_defined_fields=[
                UserDefinedField(
                    name="pressure",
                    expression="2+2",
                ),
            ],
        )

    params, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, User defined field variable name: pressure conflicts with pre-defined field names."
        " Please consider renaming this user defined field variable."
    )


def test_check_duplicate_isosurface_names():

    isosurface_qcriterion = Isosurface(name="qcriterion", field="qcriterion", iso_value=0.1)
    message = "The name `qcriterion` is reserved for the autovis isosurface from solver, please rename the isosurface."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            outputs=[IsosurfaceOutput(isosurfaces=[isosurface_qcriterion], output_fields=["Mach"])],
        )

    isosurface1 = Isosurface(name="isosurface1", field="qcriterion", iso_value=0.1)
    isosurface2 = Isosurface(name="isosurface1", field="Mach", iso_value=0.2)
    message = f"Another isosurface with name: `{isosurface2.name}` already exists, please rename the isosurface."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            outputs=[
                IsosurfaceOutput(isosurfaces=[isosurface1], output_fields=["Mach"]),
                IsosurfaceOutput(isosurfaces=[isosurface2], output_fields=["pressure"]),
            ],
        )

    message = f"Another time average isosurface with name: `{isosurface2.name}` already exists, please rename the isosurface."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(
            time_stepping=Unsteady(steps=12, step_size=0.1 * u.s),
            outputs=[
                TimeAverageIsosurfaceOutput(isosurfaces=[isosurface1], output_fields=["Mach"]),
                TimeAverageIsosurfaceOutput(isosurfaces=[isosurface2], output_fields=["pressure"]),
            ],
        )


def test_check_custom_volume_in_volume_zones():
    from flow360.component.simulation.meshing_param.volume_params import CustomZones

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
                        entities=[CustomVolume(name="zone1", boundaries=[Surface(name="face1")])],
                    ),
                    UserDefinedFarfield(),
                ],
            ),
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
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, CustomVolume zone2 is not listed under meshing->volume_zones(or zones)->CustomZones."
    )
    assert errors[0]["loc"] == ("models", 0, "entities")

    zone_3 = CustomVolume(name="zone3", boundaries=[Surface(name="face3")])
    zone_3.axis = (1, 0, 0)
    zone_3.center = (0, 0, 0) * u.mm

    with SI_unit_system:
        params = SimulationParams(
            meshing=ModularMeshingWorkflow(
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                ),
                zones=[
                    CustomZones(
                        name="custom_zones",
                        entities=[CustomVolume(name="zone1", boundaries=[Surface(name="face1")])],
                    ),
                ],
            ),
            models=[
                PorousMedium(
                    entities=[zone_2],
                    darcy_coefficient=(1, 0, 0) / u.m**2,
                    forchheimer_coefficient=(1, 0, 0) / u.m,
                    volumetric_heat_source=1.0 * u.W / u.m**3,
                ),
                Rotation(volumes=[zone_3], spec=AngularVelocity(30 * u.rpm)),
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
        validation_level="All",
    )
    assert len(errors) == 2
    assert errors[0]["msg"] == (
        "Value error, CustomVolume zone2 is not listed under meshing->volume_zones(or zones)->CustomZones."
    )
    assert errors[0]["loc"] == ("models", 0, "entities")

    assert errors[1]["msg"] == (
        "Value error, CustomVolume zone3 is not listed under meshing->volume_zones(or zones)->CustomZones."
    )
    assert errors[1]["loc"] == ("models", 1, "entities")

    zone2prim = SeedpointVolume(name="zone2", point_in_mesh=(0, 0, 0) * u.mm)
    zone2prim.axes = [(1, 0, 0), (0, 1, 0)]

    with SI_unit_system:
        params = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=2 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                ),
                zones=[
                    CustomZones(
                        entities=[SeedpointVolume(name="zone1", point_in_mesh=(0, 0, 0) * u.mm)]
                    )
                ],
            ),
            models=[
                PorousMedium(
                    entities=[zone2prim],
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
        root_item_type="Geometry",
        validation_level="All",
    )
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, SeedpointVolume zone2 is not listed under meshing->volume_zones(or zones)->CustomZones."
    )
    assert errors[0]["loc"] == ("models", 0, "entities")


def test_ghost_surface_pair_requires_quasi_3d_periodic_farfield():
    # Create two dummy ghost surfaces (Python workflow)
    periodic_1 = GhostSurface(name="periodic_1", private_attribute_id="periodic_1")
    periodic_2 = GhostSurface(name="periodic_2", private_attribute_id="periodic_2")

    # Case 1: Farfield method NOT "quasi-3d-periodic" → should raise ValueError
    with SI_unit_system, ValidationContext(CASE, quasi_3d_farfield_context), pytest.raises(
        ValueError, match="Farfield type must be 'quasi-3d-periodic' when using GhostSurfacePair."
    ):
        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational())

    # Case 2: Farfield method IS "quasi-3d-periodic" → should pass
    with SI_unit_system, ValidationContext(CASE, quasi_3d_periodic_farfield_context):
        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational())

    # Create two dummy ghost circular plane (Web UI workflow)
    periodic_1 = GhostCircularPlane(name="periodic_1", private_attribute_id="periodic_1")
    periodic_2 = GhostCircularPlane(name="periodic_2", private_attribute_id="periodic_2")

    # Case 3: Farfield method NOT "quasi-3d-periodic" → should raise ValueError
    with SI_unit_system, ValidationContext(CASE, quasi_3d_farfield_context), pytest.raises(
        ValueError, match="Farfield type must be 'quasi-3d-periodic' when using GhostSurfacePair."
    ):
        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational())

    # Case 4: Farfield method IS "quasi-3d-periodic" → should pass
    with SI_unit_system, ValidationContext(CASE, quasi_3d_periodic_farfield_context):
        Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational())


def test_seedpoint_zone_based_params():
    from flow360.component.simulation.meshing_param.volume_params import CustomZones

    with SI_unit_system:
        far_field_zone = SeedpointVolume(
            point_in_mesh=[32.5231, 112.35123, 32.342] * u.mm, name="fluid"
        )
        radiator_zone = SeedpointVolume(
            point_in_mesh=[3.2341, -1.324535, 23.345211] * u.mm,
            name="radiator",
            axes=[(1, 0, 0), (0, 1, 0)],
        )

        params = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1 * u.mm, max_spacing=100 * u.mm, gap_resolution=0.01 * u.mm
                    ),
                    smooth_controls=snappy.SmoothControls(
                        lambda_factor=0.7, mu_factor=0, iterations=3
                    ),
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(
                        boundary_layer_growth_rate=1.2,
                        boundary_layer_first_layer_thickness=0.01 * u.mm,
                    ),
                ),
                zones=[CustomZones(entities=[far_field_zone, radiator_zone])],
            ),
            operating_condition=AerospaceCondition(
                velocity_magnitude=40 * u.m / u.s,
            ),
            time_stepping=Steady(),
            models=[
                Wall(surfaces=[Surface(name="face1")]),
                PorousMedium(
                    entities=[radiator_zone],
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
        root_item_type="Geometry",
        validation_level="All",
    )

    assert errors is None


def test_deleted_surfaces_domain_type():
    # Mock Asset Cache
    surface_pos = Surface(
        name="pos_surf",
        private_attributes=SurfacePrivateAttributes(bounding_box=[[0, 1, 0], [1, 2, 1]]),
    )
    surface_neg = Surface(
        name="neg_surf",
        private_attributes=SurfacePrivateAttributes(bounding_box=[[0, -2, 0], [1, -1, 1]]),
    )
    surface_cross = Surface(
        name="cross_surf",
        private_attributes=SurfacePrivateAttributes(
            bounding_box=[[0, -0.000001, 0], [1, 0.000001, 1]]
        ),
    )

    asset_cache = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        project_entity_info=SurfaceMeshEntityInfo(
            global_bounding_box=[[0, -2, 0], [1, 2, 1]],  # Crosses Y=0
            boundaries=[surface_pos, surface_neg, surface_cross],
        ),
    )

    # Test half_body_positive_y -> keeps positive, deletes negative
    farfield = UserDefinedFarfield(domain_type="half_body_positive_y")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    planar_face_tolerance=1e-4,
                    geometry_accuracy=1e-5,
                    boundary_layer_first_layer_thickness=1e-3,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(entities=[surface_pos]),  # OK
                Wall(entities=[surface_neg]),  # Error
                Wall(entities=[surface_cross]),  # OK (touches 0)
            ],
            private_attribute_asset_cache=asset_cache,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    assert len(errors) == 1
    assert "Boundary `neg_surf` will likely be deleted" in errors[0]["msg"]

    # Test half_body_negative_y -> keeps negative, deletes positive
    farfield_neg = UserDefinedFarfield(domain_type="half_body_negative_y")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    planar_face_tolerance=1e-4,
                    geometry_accuracy=1e-5,
                    boundary_layer_first_layer_thickness=1e-3,
                ),
                volume_zones=[farfield_neg],
            ),
            models=[
                Wall(entities=[surface_pos]),  # Error
                Wall(entities=[surface_neg]),  # OK
                Wall(entities=[surface_cross]),  # OK
            ],
            private_attribute_asset_cache=asset_cache,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    assert len(errors) == 1
    assert "Boundary `pos_surf` will likely be deleted" in errors[0]["msg"]


def test_unique_selector_names():
    """Test that duplicate selector names are detected and raise an error."""
    from flow360.component.simulation.framework.entity_selector import (
        SurfaceSelector,
        collect_and_tokenize_selectors_in_place,
    )
    from flow360.component.simulation.models.surface_models import Wall
    from flow360.component.simulation.primitives import Surface

    # Create actual Surface entities to avoid selector expansion issues
    surface1 = Surface(name="surface1")
    surface2 = Surface(name="surface2")

    # Create selectors with duplicate names
    selector1 = SurfaceSelector(name="duplicate_name").match("wing*")
    selector2 = SurfaceSelector(name="duplicate_name").match("tail*")

    # Test duplicate selector names in different EntityLists (different Wall models)
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Wall(entities=[surface1, selector1]),
                Wall(entities=[surface2, selector2]),
            ],
        )

    # Tokenize selectors to populate used_selectors (simulating what happens in set_up_params_for_uploading)
    params_dict = params.model_dump(mode="json", exclude_none=True)
    params_dict = collect_and_tokenize_selectors_in_place(params_dict)

    # Now validate using validate_model which will materialize and validate
    _, errors, _ = validate_model(
        params_as_dict=params_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    assert errors is not None
    assert len(errors) == 1
    assert "Duplicate selector name 'duplicate_name'" in errors[0]["msg"]

    # Test duplicate selector names in the same EntityList
    with SI_unit_system:
        params2 = SimulationParams(
            models=[
                Wall(entities=[surface1, selector1, selector2]),
            ],
        )

    params_dict2 = params2.model_dump(mode="json", exclude_none=True)
    params_dict2 = collect_and_tokenize_selectors_in_place(params_dict2)

    _, errors2, _ = validate_model(
        params_as_dict=params_dict2,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    assert errors2 is not None
    assert len(errors2) == 1
    assert "Duplicate selector name 'duplicate_name'" in errors2[0]["msg"]

    # Test that unique selector names work fine
    selector3 = SurfaceSelector(name="unique_name_1").match("wing*")
    selector4 = SurfaceSelector(name="unique_name_2").match("tail*")

    with SI_unit_system:
        params3 = SimulationParams(
            models=[
                Wall(entities=[surface1, selector3]),
                Wall(entities=[surface2, selector4]),
            ],
        )

    params_dict3 = params3.model_dump(mode="json", exclude_none=True)
    params_dict3 = collect_and_tokenize_selectors_in_place(params_dict3)

    validated_params, errors3, _ = validate_model(
        params_as_dict=params_dict3,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    # Should not have errors for unique names
    assert errors3 is None or len(errors3) == 0
    assert validated_params is not None


def test_coordinate_system_requires_geometry_ai():
    """Test that CoordinateSystem is only supported when Geometry AI is enabled."""
    # Create a CoordinateSystemStatus with assignments
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(entity_type="GeometryBodyGroup", entity_id="test-id")
                ],
            )
        ],
    )

    # Asset cache with GAI disabled but coordinate system used
    asset_cache_no_gai = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )

    with SI_unit_system:
        params = SimulationParams(
            private_attribute_asset_cache=asset_cache_no_gai,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    assert errors is not None
    assert any(
        "Coordinate system assignment to GeometryBodyGroup" in str(e)
        and "Geometry AI is enabled" in str(e)
        for e in errors
    )

    # Test with GAI enabled - should pass
    asset_cache_with_gai = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        coordinate_system_status=cs_status,
    )

    with SI_unit_system:
        params = SimulationParams(
            private_attribute_asset_cache=asset_cache_with_gai,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    # No error about coordinate system
    assert errors is None or not any("CoordinateSystem" in str(e) for e in errors)


def test_mirroring_requires_geometry_ai():
    """Test that mirroring is only supported when Geometry AI is enabled."""
    # Create a MirrorStatus with mirrored entities
    mirror_plane = MirrorPlane(
        name="test_plane",
        normal=(0, 1, 0),
        center=[0, 0, 0] * u.m,
    )
    mirrored_group = MirroredGeometryBodyGroup(
        name="test_<mirror>",
        geometry_body_group_id="test-body-group",
        mirror_plane_id=mirror_plane.private_attribute_id,
    )
    mirror_status = MirrorStatus(
        mirror_planes=[mirror_plane],
        mirrored_geometry_body_groups=[mirrored_group],
        mirrored_surfaces=[],
    )

    # Asset cache with GAI disabled but mirroring used
    asset_cache_no_gai = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=False,
        mirror_status=mirror_status,
    )

    with SI_unit_system:
        params = SimulationParams(
            private_attribute_asset_cache=asset_cache_no_gai,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    assert errors is not None
    assert any("Mirroring is only supported when Geometry AI is enabled" in str(e) for e in errors)

    # Test with GAI enabled - should pass
    asset_cache_with_gai = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        mirror_status=mirror_status,
    )

    with SI_unit_system:
        params = SimulationParams(
            private_attribute_asset_cache=asset_cache_with_gai,
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    # No error about mirroring
    assert errors is None or not any("Mirroring" in str(e) for e in errors)


def test_mirror_missing_boundary_condition_downgraded_to_warning():
    """Missing BCs should be downgraded to warnings when mirroring/transformations are detected."""
    mirror_plane = MirrorPlane(
        name="test_plane",
        normal=(0, 1, 0),
        center=[0, 0, 0] * u.m,
        private_attribute_id="mp-1",
    )

    front = Surface(name="front", private_attribute_is_interface=False, private_attribute_id="s-1")
    mirrored_front = MirroredSurface(
        name="front_<mirror>",
        surface_id="s-1",
        mirror_plane_id="mp-1",
        private_attribute_id="ms-1",
    )

    asset_cache = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        project_entity_info=VolumeMeshEntityInfo(boundaries=[front]),
        mirror_status=MirrorStatus(
            mirror_planes=[mirror_plane],
            mirrored_geometry_body_groups=[],
            mirrored_surfaces=[mirrored_front],
        ),
    )

    with SI_unit_system:
        params = SimulationParams(
            models=[Fluid(), Wall(entities=[front])],
            private_attribute_asset_cache=asset_cache,
        )

    _validated, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    assert errors is None
    assert any("front_<mirror>" in w.get("msg", "") for w in warnings), warnings


def test_mirror_unknown_boundary_still_raises_error():
    """Unknown boundary names should remain hard errors even when mirroring is detected."""
    mirror_plane = MirrorPlane(
        name="test_plane",
        normal=(0, 1, 0),
        center=[0, 0, 0] * u.m,
        private_attribute_id="mp-1",
    )

    front = Surface(name="front", private_attribute_is_interface=False, private_attribute_id="s-1")
    mirrored_front = MirroredSurface(
        name="front_<mirror>",
        surface_id="s-1",
        mirror_plane_id="mp-1",
        private_attribute_id="ms-1",
    )

    asset_cache = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        project_entity_info=VolumeMeshEntityInfo(boundaries=[front]),
        mirror_status=MirrorStatus(
            mirror_planes=[mirror_plane],
            mirrored_geometry_body_groups=[],
            mirrored_surfaces=[mirrored_front],
        ),
    )

    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(),
                # Use mirrored surface (should be known once we include it in the valid boundary pool)
                Wall(entities=[mirrored_front, Surface(name="typo_surface")]),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    _validated, errors, _warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    assert errors is not None
    assert any("typo_surface" in str(e) for e in errors)


def test_domain_type_bbox_mismatch_downgraded_to_warning_when_transformed():
    """domain_type bbox mismatch should be a warning when transformations are detected."""
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[CoordinateSystem(name="cs", private_attribute_id="cs-1")],
        parents=[],
        assignments=[CoordinateSystemAssignmentGroup(coordinate_system_id="cs-1", entities=[])],
    )

    # Global bbox fully on -Y side; choosing half_body_positive_y should normally raise.
    asset_cache = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        project_entity_info=SurfaceMeshEntityInfo(
            boundaries=[],
            global_bounding_box=[[-1, -10, -1], [1, -5, 1]],
        ),
        coordinate_system_status=cs_status,
    )

    auto_farfield = AutomatedFarfield(name="my_farfield", domain_type="half_body_positive_y")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    geometry_accuracy=1e-10 * u.m,
                    surface_max_edge_length=1e-10,
                ),
                volume_zones=[auto_farfield],
            ),
            private_attribute_asset_cache=asset_cache,
        )

    _validated, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )

    assert errors is None
    assert any(
        "domain_type" in w.get("msg", "") or "symmetry plane" in w.get("msg", "") for w in warnings
    ), warnings


def test_incomplete_BC_with_geometry_AI():
    """Test that missing boundary conditions produce warnings (not errors) when using Geometry AI."""
    # Construct a dummy asset cache with GAI enabled
    wall = Surface(name="wall", private_attribute_is_interface=False, private_attribute_id="wall")
    no_bc = Surface(
        name="no_bc", private_attribute_is_interface=False, private_attribute_id="no_bc"
    )

    asset_cache = AssetCache(
        project_length_unit="m",
        project_entity_info=VolumeMeshEntityInfo(boundaries=[wall, no_bc]),
        use_geometry_AI=True,  # Enable GAI
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                )
            ),
            models=[
                Fluid(),
                Wall(entities=[wall]),
                # no_bc is intentionally missing
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    # Should not have errors, only warnings
    assert errors is None or len(errors) == 0
    assert len(warnings) > 0
    assert any(
        "no_bc" in w.get("msg", "") and "do not have a boundary condition" in w.get("msg", "")
        for w in warnings
    ), f"Expected warning about missing boundary condition for 'no_bc', got: {warnings}"


def test_incomplete_BC_without_geometry_AI():
    """Test that missing boundary conditions produce errors when NOT using Geometry AI."""
    # Construct a dummy asset cache without GAI
    wall = Surface(name="wall", private_attribute_is_interface=False, private_attribute_id="wall")
    no_bc = Surface(
        name="no_bc", private_attribute_is_interface=False, private_attribute_id="no_bc"
    )

    asset_cache = AssetCache(
        project_length_unit="m",
        project_entity_info=VolumeMeshEntityInfo(boundaries=[wall, no_bc]),
        use_geometry_AI=False,  # Disable GAI
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-10,
                    surface_max_edge_length=1e-10,
                )
            ),
            models=[
                Fluid(),
                Wall(entities=[wall]),
                # no_bc is intentionally missing
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    # Should have errors
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, The following boundaries do not have a boundary condition: no_bc. "
        "Please add them to a boundary condition model in the `models` section."
    )
