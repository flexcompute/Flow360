import json
import os
import re

import pytest
import unyt as u

import flow360.component.simulation.user_code.core.context as context
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    NoneSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import (
    AngularVelocity,
    Fluid,
    Rotation,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    GenericReferenceCondition,
    LiquidOperatingCondition,
)
from flow360.component.simulation.outputs.output_entities import Isosurface
from flow360.component.simulation.outputs.outputs import IsosurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    initialize_variable_space,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    get_referenced_expressions_and_user_variables,
    save_user_variables,
)
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import control, solution
from flow360.component.volume_mesh import VolumeMeshV2


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def reset_context():
    """Clear user variables from the context."""
    clear_context()


def volume_mesh():
    return VolumeMeshV2.from_local_storage(
        mesh_id=None,
        local_storage_path=os.path.join(os.path.dirname(__file__), "data", "vm_entity_provider"),
    )


def asset_cache():
    with open(
        os.path.join(os.path.dirname(__file__), "data", "vm_entity_provider", "simulation.json"),
        "r",
    ) as fh:
        asset_cache_data = json.load(fh).pop("private_attribute_asset_cache")
    return AssetCache.model_validate(asset_cache_data)


def operating_condition_with_expression():
    reset_context()
    vm = volume_mesh()
    vel = UserVariable(name="my_speed", value=3.6 * u.km / u.hr)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=vel * 314),
            models=[Wall(name="wall", entities=vm["*"])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def liquid_operating_condition_with_expression():
    reset_context()
    vm = volume_mesh()
    vel = UserVariable(name="my_speed_2", value=600 * u.m / u.min)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=vel),
            models=[Wall(name="wall", entities=vm["*"])],
            time_stepping=Unsteady(step_size=1 * u.s, steps=10),
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def generic_operating_condition_with_expression():
    reset_context()
    vm = volume_mesh()
    vel = UserVariable(name="my_speed_3", value=314 * u.km / u.hr)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=GenericReferenceCondition(velocity_magnitude=vel * 3.6),
            models=[Wall(name="wall", entities=vm["*"])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def reference_area_with_expression():
    reset_context()
    vm = volume_mesh()
    var = UserVariable(name="area_var", value=(150 * u.cm) ** 2)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            reference_geometry=ReferenceGeometry(area=var),
            models=[Wall(name="wall", entities=vm["*"])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def angular_velocity_with_expression():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    var = UserVariable(name="angular_velocity_var", value=math.sqrt(0.01) * u.rad / u.s)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Wall(name="wall", entities=vm["*"]),
                Rotation(name="rotation", entities=vm["*"], spec=AngularVelocity(value=var)),
            ],
            time_stepping=Unsteady(step_size=1 * u.s, steps=10),
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def time_stepping_with_expression():
    reset_context()
    vm = volume_mesh()
    var = UserVariable(name="time_step_size_var", value=math.sqrt(100) * u.s)
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[Wall(name="wall", entities=vm["*"])],
            time_stepping=Unsteady(step_size=var, steps=10),
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    "param_dict, ref_dict_path",
    [
        (
            operating_condition_with_expression(),
            "ref/value_or_expression/op_vel_mag.json",
        ),
        (
            liquid_operating_condition_with_expression(),
            "ref/value_or_expression/liquid_op_vel_mag.json",
        ),
        (
            generic_operating_condition_with_expression(),
            "ref/value_or_expression/op_vel_mag.json",
        ),
        (
            reference_area_with_expression(),
            "ref/value_or_expression/ref_area_with_expression.json",
        ),
        (
            angular_velocity_with_expression(),
            "ref/value_or_expression/angular_velocity_with_expression.json",
        ),
        (
            time_stepping_with_expression(),
            "ref/value_or_expression/time_stepping_with_expression.json",
        ),
    ],
)
def test_e2e_dump_validate_and_translate(param_dict: dict, ref_dict_path: str):
    params, errors, _ = validate_model(
        params_as_dict=param_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )
    assert errors is None, "Errors: {errors}"
    translated = get_solver_json(params, mesh_unit=10 * u.m)
    try:
        with open(ref_dict_path, "r") as fh:
            ref_dict = json.load(fh)
        assert compare_values(ref_dict, translated)
    except FileNotFoundError as e:
        print("=======\n", json.dumps(translated, indent=2), "\n=======")
        raise e


def param_with_SST():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=KOmegaSST()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[solution.nu_hat])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_SpalartAllmaras():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[
                VolumeOutput(name="output", output_fields=[solution.turbulence_kinetic_energy])
            ],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_SpalartAllmaras_specific_dissipation():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[
                VolumeOutput(name="output", output_fields=[solution.specific_rate_of_dissipation])
            ],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_without_transition_model():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(
                    turbulence_model_solver=SpalartAllmaras(),
                    transition_model_solver=NoneSolver(),
                ),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[solution.amplification_factor])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_without_transition_model_intermittency():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(
                    turbulence_model_solver=SpalartAllmaras(),
                    transition_model_solver=NoneSolver(),
                ),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[
                VolumeOutput(name="output", output_fields=[solution.turbulence_intermittency])
            ],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_liquid_operating_condition_density():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[solution.density])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_liquid_operating_condition_temperature():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[solution.temperature])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_liquid_operating_condition_mach():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(velocity_magnitude=10 * u.m / u.s),
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[solution.Mach])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_steady_time_stepping_physical_step():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[control.physicalStep])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_steady_time_stepping_time_step_size():
    reset_context()
    vm = volume_mesh()
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[control.timeStepSize])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_rotation_zone_theta():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
                Rotation(
                    name="rotation", entities=vm["fluid"], spec=AngularVelocity(value=100 * u.rpm)
                ),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[control.theta])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_rotation_zone_omega():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
                Rotation(
                    name="rotation", entities=vm["fluid"], spec=AngularVelocity(value=100 * u.rpm)
                ),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[control.omega])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


def param_with_rotation_zone_omega_dot():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
                Rotation(
                    name="rotation", entities=vm["fluid"], spec=AngularVelocity(value=100 * u.rpm)
                ),
            ],
            outputs=[VolumeOutput(name="output", output_fields=[control.omegaDot])],
            private_attribute_asset_cache=asset_cache(),
        )
    return save_user_variables(params).model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    "param_as_dict, expected_error_msg",
    [
        (
            param_with_SST(),
            "`solution.nu_hat` cannot be used because Spalart-Allmaras turbulence solver is not used.",
        ),
        (
            param_with_SpalartAllmaras(),
            "`solution.turbulence_kinetic_energy` cannot be used because k-omega turbulence solver is not used.",
        ),
        (
            param_with_SpalartAllmaras_specific_dissipation(),
            "`solution.specific_rate_of_dissipation` cannot be used because k-omega turbulence solver is not used.",
        ),
        (
            param_without_transition_model(),
            "`solution.amplification_factor` cannot be used because Amplification factor transition model is not used.",
        ),
        (
            param_without_transition_model_intermittency(),
            "`solution.turbulence_intermittency` cannot be used because Amplification factor transition model is not used.",
        ),
        (
            param_with_liquid_operating_condition_density(),
            "`solution.density` cannot be used because Liquid operating condition is used.",
        ),
        (
            param_with_liquid_operating_condition_temperature(),
            "`solution.temperature` cannot be used because Liquid operating condition is used.",
        ),
        (
            param_with_liquid_operating_condition_mach(),
            "`solution.Mach` cannot be used because Liquid operating condition is used.",
        ),
        (
            param_with_steady_time_stepping_physical_step(),
            "`control.physicalStep` cannot be used because Unsteady time stepping is not used.",
        ),
        (
            param_with_steady_time_stepping_time_step_size(),
            "`control.timeStepSize` cannot be used because Unsteady time stepping is not used.",
        ),
        (
            param_with_rotation_zone_theta(),
            "`control.theta` cannot be used because Rotation zone is not used.",
        ),
        (
            param_with_rotation_zone_omega(),
            "`control.omega` cannot be used because Rotation zone is not used.",
        ),
        (
            param_with_rotation_zone_omega_dot(),
            "`control.omegaDot` cannot be used because Rotation zone is not used.",
        ),
    ],
)
def test_feature_requirement_map(param_as_dict: dict, expected_error_msg: str):
    """Test feature requirement map."""
    _, errors, _ = validate_model(
        params_as_dict=param_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert len(errors) == 1
    assert expected_error_msg in errors[0]["msg"]


def test_get_referenced_expressions():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    UserVariable(name="unused_var", value=100 * u.rpm)
    UserVariable(name="unused_var2", value=solution.turbulence_kinetic_energy)  # Should not raise
    fv = UserVariable(
        name="forbidden_var", value=solution.specific_rate_of_dissipation
    )  # Should raise (convoluted)
    uv3 = UserVariable(name="used_var3", value=fv + 12 / u.hr)

    var = UserVariable(name="vel_mag", value=math.magnitude(solution.velocity) - 5 * u.cm / u.ms)
    iso_field = UserVariable(name="iso_field", value=var + 123 * u.m / u.s - uv3 * 2 * u.cm)
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            outputs=[
                VolumeOutput(name="vol_output", output_fields=[solution.velocity, iso_field]),
                IsosurfaceOutput(
                    name="iso_surface",
                    entities=[
                        Isosurface(name="iso_surface", field=iso_field, iso_value=1200 * u.m / u.s)
                    ],
                    output_fields=[solution.Mach],
                ),
            ],
            time_stepping=Unsteady(step_size=Expression(expression="(123 - 5) * u.s"), steps=10),
            private_attribute_asset_cache=asset_cache(),
        )
    param_as_dict = save_user_variables(params).model_dump(mode="json", exclude_none=True)

    context_var_names = [
        item["name"] for item in param_as_dict["private_attribute_asset_cache"]["variable_context"]
    ]
    assert "unused_var" in context_var_names
    assert "unused_var2" in context_var_names

    initialize_variable_space(param_as_dict, is_clear_context=True)
    expressions = get_referenced_expressions_and_user_variables(param_as_dict)

    assert sorted(expressions) == [
        "(123 - 5) * u.s",
        "forbidden_var + 12 * 1 / u.hr",
        "math.magnitude(solution.velocity) - 5 * u.cm / u.ms",
        "solution.Mach",
        "solution.specific_rate_of_dissipation",
        "solution.velocity",
        "vel_mag + 123 * u.m / u.s - used_var3 * 2 * u.cm",
    ], f"Expressions: {expressions}"

    _, error, _ = validate_model(
        params_as_dict=param_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert len(error) == 1
    assert error[0]["loc"] == ("private_attribute_asset_cache", "variable_context", 2, "value")
    assert (
        error[0]["msg"]
        == "Value error, `solution.specific_rate_of_dissipation` cannot be used because k-omega turbulence solver is not used."
    )


def test_integer_validation():
    with SI_unit_system:
        AerospaceCondition(velocity_magnitude=10)

    with pytest.raises(
        ValueError,
        match=re.escape("Value error, arg '10' does not match (length)/(time) dimension."),
    ):
        with SI_unit_system:
            AerospaceCondition(velocity_magnitude=Expression(expression="10"))


def test_param_with_number_expression_in_and_out():
    reset_context()
    vm = volume_mesh()
    vm["fluid"].axis = (0, 1, 0)
    vm["fluid"].center = (1, 1, 2) * u.cm
    my_step_size = UserVariable(name="my_step_size", value=[9, 38 + 2, 0] * u.s)
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Fluid(turbulence_model_solver=SpalartAllmaras()),
                Wall(name="wall", entities=vm["*"]),
            ],
            time_stepping=Unsteady(step_size=math.magnitude(my_step_size), steps=10),
            private_attribute_asset_cache=asset_cache(),
        )
    processed_params: SimulationParams = save_user_variables(params)
    assert (
        processed_params.private_attribute_asset_cache.variable_context[0].value.expression
        == "[9, 40, 0] * u.s"
    )
    params_as_dict = processed_params.model_dump(mode="json", exclude_none=True)
    reset_context()
    new_params, _, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="Case",
    )

    assert new_params.time_stepping.step_size.evaluate() == 41 * u.s
