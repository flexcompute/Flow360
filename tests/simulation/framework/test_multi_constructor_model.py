import os
from copy import deepcopy

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock
from tests.simulation.converter.test_bet_translator import generate_BET_param


@pytest.fixture
def get_aerospace_condition_default():
    return AerospaceCondition(
        velocity_magnitude=0.8 * u.km / u.s,
        alpha=5 * u.deg,
        thermal_state=ThermalState(),
        reference_velocity_magnitude=100 * u.m / u.s,
    )


@pytest.fixture
def get_aerospace_condition_default_and_thermal_state_using_from():
    return AerospaceCondition(
        velocity_magnitude=0.8 * u.km / u.s,
        alpha=5 * u.deg,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000 * u.m),
        reference_velocity_magnitude=100 * u.m / u.s,
    )


@pytest.fixture
def get_aerospace_condition_using_from():
    return AerospaceCondition.from_mach(
        mach=0.8,
        alpha=5 * u.deg,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000 * u.m),
    )


<<<<<<< HEAD
=======
@pytest.fixture
def get_aerospace_condition_using_from_mach_reynolds():
    return AerospaceCondition.from_mach_reynolds(
        mach=0.8,
        reynolds_mesh_unit=1e6,
        project_length_unit=u.m,
        alpha=5 * u.deg,
        temperature=290 * u.K,
    )


>>>>>>> f5627a46 (changed mach reynolds to use charcteristic length (#1138))
def compare_objects_from_dict(dict1: dict, dict2: dict, object_class: type[Flow360BaseModel]):
    obj1 = object_class.model_validate(dict1)
    obj2 = object_class.model_validate(dict2)
    assert obj1.model_dump_json() == obj2.model_dump_json()


def test_full_model(get_aerospace_condition_default, get_aerospace_condition_using_from):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=False)
    data_parsed = parse_model_dict(full_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from.model_dump(exclude_none=False)
    data_parsed = parse_model_dict(full_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)


def test_incomplete_model(
    get_aerospace_condition_default,
    get_aerospace_condition_using_from,
    get_aerospace_condition_default_and_thermal_state_using_from,
):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=False)
    incomplete_data = deepcopy(full_data)
    incomplete_data["private_attribute_input_cache"] = {}
    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from.model_dump(exclude_none=False)
    incomplete_data = {
        "type_name": full_data["type_name"],
        "private_attribute_constructor": full_data["private_attribute_constructor"],
        "private_attribute_input_cache": full_data["private_attribute_input_cache"],
    }

    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_default_and_thermal_state_using_from.model_dump(
        exclude_none=False
    )
    incomplete_data = deepcopy(full_data)
    incomplete_data["thermal_state"] = {
        "type_name": full_data["thermal_state"]["type_name"],
        "private_attribute_constructor": full_data["thermal_state"][
            "private_attribute_constructor"
        ],
        "private_attribute_input_cache": full_data["thermal_state"][
            "private_attribute_input_cache"
        ],
    }

    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)


def test_recursive_incomplete_model(get_aerospace_condition_using_from):
    # `incomplete_data` contains only the private_attribute_* for both the AerospaceCondition and ThermalState
    full_data = get_aerospace_condition_using_from.model_dump(exclude_none=False)
    input_cache = full_data["private_attribute_input_cache"]
    input_cache["thermal_state"] = {
        "type_name": input_cache["thermal_state"]["type_name"],
        "private_attribute_constructor": input_cache["thermal_state"][
            "private_attribute_constructor"
        ],
        "private_attribute_input_cache": input_cache["thermal_state"][
            "private_attribute_input_cache"
        ],
    }
    incomplete_data = {
        "type_name": full_data["type_name"],
        "private_attribute_constructor": full_data["private_attribute_constructor"],
        "private_attribute_input_cache": full_data["private_attribute_input_cache"],
    }

    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)


def test_entity_with_multi_constructor():

    class ModelWithEntityList(Flow360BaseModel):
        entities: EntityList[Box, Cylinder] = pd.Field()

    with SI_unit_system:
        model = ModelWithEntityList(
            entities=[
                Box(
                    name="my_box_default",
                    center=(1, 2, 3),
                    size=(2, 2, 3),
                    angle_of_rotation=20 * u.deg,
                    axis_of_rotation=(1, 0, 0),
                ),
                Box.from_principal_axes(
                    name="my_box_from",
                    center=(7, 1, 2),
                    size=(2, 2, 3),
                    axes=((3 / 5, 4 / 5, 0), (4 / 5, -3 / 5, 0)),
                ),
                Cylinder(
                    name="my_cylinder_default",
                    axis=(0, 1, 0),
                    center=(1, 2, 3),
                    outer_radius=2,
                    height=3,
                ),
            ]
        )
    full_data = model.model_dump(exclude_none=False)
    incomplete_data = {"entities": {"stored_entities": []}}
    # For default constructed entity we do not do anything
    incomplete_data["entities"]["stored_entities"].append(
        full_data["entities"]["stored_entities"][0]
    )
    incomplete_data["entities"]["stored_entities"][0]["private_attribute_input_cache"] = {}
    entity_dict = full_data["entities"]["stored_entities"][1]
    incomplete_entity = {}
    for key, value in entity_dict.items():
        if key in [
            "type_name",
            "private_attribute_constructor",
            "private_attribute_input_cache",
            "private_attribute_id",
        ]:
            incomplete_entity[key] = value
    incomplete_data["entities"]["stored_entities"].append(incomplete_entity)
    incomplete_data["entities"]["stored_entities"].append(
        full_data["entities"]["stored_entities"][2]
    )

    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, ModelWithEntityList)


def test_entity_modification(get_aerospace_condition_using_from):

    my_box = Box.from_principal_axes(
        name="box",
        axes=[(0, 1, 0), (0, 0, 1)],
        center=(0, 0, 0) * u.m,
        size=(0.2, 0.3, 2) * u.m,
    )

    my_box.center = (1, 2, 3) * u.m
    assert all(my_box.private_attribute_input_cache.center == (1, 2, 3) * u.m)

    my_box = Box(
        name="box2",
        axis_of_rotation=(1, 0, 0),
        angle_of_rotation=45 * u.deg,
        center=(1, 1, 1) * u.m,
        size=(0.2, 0.3, 2) * u.m,
    )

    my_box.size = (1, 2, 32) * u.m
    assert all(my_box.private_attribute_input_cache.size == (1, 2, 32) * u.m)

    my_op = get_aerospace_condition_using_from
    my_op.alpha = -12 * u.rad
    assert my_op.private_attribute_input_cache.alpha == -12 * u.rad


def test_BETDisk_multi_constructor_full():
    for bet_type in ["c81", "dfdc", "xfoil", "xrotor"]:
        bet = generate_BET_param(bet_type)
        full_data = bet.model_dump(exclude_none=False)
        data_parsed = parse_model_dict(full_data, globals())
        compare_objects_from_dict(full_data, data_parsed, BETDisk)


def test_BETDisk_multi_constructor_cache_only():
    for bet_type in ["c81", "dfdc", "xfoil", "xrotor"]:
        original_workdir = os.getcwd()
        try:
            # Mimicking customer using a relative path for the files.
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            bet = generate_BET_param(bet_type, given_path_prefix="../converter/")
        finally:
            # Ooops I changed my directory (trying using the json in some other folder)
            os.chdir(original_workdir)

        full_data = bet.model_dump(exclude_none=False)
        incomplete_data = {
            "type_name": full_data["type_name"],
            "private_attribute_constructor": full_data["private_attribute_constructor"],
            "private_attribute_input_cache": full_data["private_attribute_input_cache"],
        }
        # Make sure cache only can be deserialized and that we won't have
        # trouble even if we switch directory where the file path no longer is valid.
        data_parsed = parse_model_dict(incomplete_data, globals())
        compare_objects_from_dict(full_data, data_parsed, BETDisk)
