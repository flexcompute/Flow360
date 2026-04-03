import os
from copy import deepcopy

import pytest
from flow360_schema.framework.validation.context import DeserializationContext

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
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
def get_aerospace_condition_using_from_mach():
    return AerospaceCondition.from_mach(
        mach=0.8,
        alpha=5 * u.deg,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000 * u.m),
    )


@pytest.fixture
def get_aerospace_condition_using_from_mach_reynolds():
    return AerospaceCondition.from_mach_reynolds(
        mach=0.8,
        reynolds_mesh_unit=1e6,
        project_length_unit=u.m,
        alpha=5 * u.deg,
        temperature=290 * u.K,
    )


def compare_objects_from_dict(dict1: dict, dict2: dict, object_class: type[Flow360BaseModel]):
    with DeserializationContext():
        obj1 = object_class.model_validate(dict1)
        obj2 = object_class.model_validate(dict2)
    assert obj1.model_dump_json() == obj2.model_dump_json()


def test_full_model(
    get_aerospace_condition_default,
    get_aerospace_condition_using_from_mach,
    get_aerospace_condition_using_from_mach_reynolds,
):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=False)
    data_parsed = parse_model_dict(full_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from_mach.model_dump(exclude_none=False)
    data_parsed = parse_model_dict(full_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from_mach_reynolds.model_dump(exclude_none=False)
    data_parsed = parse_model_dict(full_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)


def test_incomplete_model(
    get_aerospace_condition_default,
    get_aerospace_condition_using_from_mach,
    get_aerospace_condition_using_from_mach_reynolds,
    get_aerospace_condition_default_and_thermal_state_using_from,
):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=False)
    incomplete_data = deepcopy(full_data)
    incomplete_data["private_attribute_input_cache"] = {}
    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from_mach.model_dump(exclude_none=False)
    incomplete_data = {
        "type_name": full_data["type_name"],
        "private_attribute_constructor": full_data["private_attribute_constructor"],
        "private_attribute_input_cache": full_data["private_attribute_input_cache"],
    }

    data_parsed = parse_model_dict(incomplete_data, globals())
    compare_objects_from_dict(full_data, data_parsed, AerospaceCondition)

    full_data = get_aerospace_condition_using_from_mach_reynolds.model_dump(exclude_none=False)
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


def test_recursive_incomplete_model(get_aerospace_condition_using_from_mach):
    # `incomplete_data` contains only the private_attribute_* for both the AerospaceCondition and ThermalState
    full_data = get_aerospace_condition_using_from_mach.model_dump(exclude_none=False)
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


def test_non_entity_modification_updates_input_cache(get_aerospace_condition_using_from_mach):
    my_op = get_aerospace_condition_using_from_mach
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
            "private_attribute_id": full_data["private_attribute_id"],
        }
        # Make sure cache only can be deserialized and that we won't have
        # trouble even if we switch directory where the file path no longer is valid.
        data_parsed = parse_model_dict(incomplete_data, globals())
        compare_objects_from_dict(full_data, data_parsed, BETDisk)
