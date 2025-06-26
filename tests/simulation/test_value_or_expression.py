import json
import os

import pytest
import unyt as u

from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import AngularVelocity, Rotation
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    GenericReferenceCondition,
    LiquidOperatingCondition,
)
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import (
    UserVariable,
    default_context,
    save_user_variables,
)
from flow360.component.simulation.user_code.functions import math
from flow360.component.volume_mesh import VolumeMeshV2


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


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
    default_context.clear()
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
    default_context.clear()
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
    default_context.clear()
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
    default_context.clear()
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
    default_context.clear()
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
    default_context.clear()
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
