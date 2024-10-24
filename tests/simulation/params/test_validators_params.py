import json
import re
import unittest
from typing import Literal

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.material import SolidMaterial, aluminum
from flow360.component.simulation.models.surface_models import (
    Freestream,
    Periodic,
    SlipWall,
    Translational,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    Fluid,
    HeatEquationInitialCondition,
    NavierStokesInitialCondition,
    Solid,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import GenericVolume, Surface
from flow360.component.simulation.services import validate_model
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
def volume_output_with_SA_DDES():
    volume_output = VolumeOutput(name="volume", output_fields=["SpalartAllmaras_DDES"])
    return volume_output


@pytest.fixture()
def volume_output_with_kOmega_DDES():
    volume_output = VolumeOutput(name="volume", output_fields=["kOmegaSST_DDES"])
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
def fluid_model_with_DDES():
    fluid_model = Fluid()
    fluid_model.turbulence_model_solver.DDES = True
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


def test_ddes_wall_function_validator(
    volume_output_with_SA_DDES,
    volume_output_with_kOmega_DDES,
    fluid_model_with_DDES,
    fluid_model,
):
    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[fluid_model_with_DDES], outputs=[volume_output_with_SA_DDES]
        )

    assert params

    message = "kOmegaSST_DDES output can only be specified with kOmegaSST turbulence model and DDES turned on."

    # Invalid simulation params (wrong output type)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(models=[fluid_model_with_DDES], outputs=[volume_output_with_kOmega_DDES])

    # Invalid simulation params (DDES turned off)
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SimulationParams(models=[fluid_model], outputs=[volume_output_with_kOmega_DDES])


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

    message = "In `Solid` model -> material, the heat capacity needs to be specified for unsteady simulations."

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model, solid_model_without_specific_heat_capacity],
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

    message = "In `Solid` model, the initial condition needs to be specified "
    "when the `Fluid` model uses expression as initial condition."

    # Invalid simulation params
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        _ = SimulationParams(
            models=[fluid_model_with_initial_condition, solid_model_without_initial_condition],
            time_stepping=Steady(),
            outputs=[surface_output_with_residual_heat_solver],
        )


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

    with SI_unit_system:
        param = SimulationParams(
            models=[
                Fluid(),
                Wall(entities=wall_1),
                Periodic(surface_pairs=(periodic_1, periodic_2), spec=Translational()),
                SlipWall(entities=[i_exist]),
                Freestream(entities=auto_farfield.farfield),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    def _validate_under_CASE(param):
        param_dict = json.loads(param.model_dump_json())
        with ValidationLevelContext(CASE):
            with SI_unit_system:
                SimulationParams.model_validate(param_dict)

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"no_bc does not have a boundary condition. Please add to it a model in the `models` section."
        ),
    ):
        _validate_under_CASE(param)

    param.models.append(SlipWall(entities=[Surface(name="plz_dont_do_this"), no_bc]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"plz_dont_do_this is not a known `Surface` entity but it appears in the `models` section."
        ),
    ):
        _validate_under_CASE(param)


def test_duplicate_entities_in_models():
    entity_generic_volume = GenericVolume(name="Duplicate Volume")
    entity_surface = Surface(name="Duplicate Surface")
    volume_model1 = Solid(
        volumes=[entity_generic_volume],
        material=aluminum,
        volumetric_heat_source="0",
    )
    volume_model2 = volume_model1
    surface_model1 = SlipWall(entities=[entity_surface])
    surface_model2 = Wall(entities=[entity_surface])
    surface_model3 = surface_model1

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[volume_model1, surface_model1],
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
