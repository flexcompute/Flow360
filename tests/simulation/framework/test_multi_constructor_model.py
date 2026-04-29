import os

from flow360_schema.framework.validation.context import DeserializationContext

from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.models.volume_models import BETDisk
from tests.simulation.converter.test_bet_translator import generate_BET_param


def compare_objects_from_dict(dict1, dict2, object_class):
    with DeserializationContext():
        obj1 = object_class.model_validate(dict1)
        obj2 = object_class.model_validate(dict2)
    assert obj1.model_dump_json() == obj2.model_dump_json()


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
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            bet = generate_BET_param(bet_type, given_path_prefix="../converter/")
        finally:
            os.chdir(original_workdir)

        full_data = bet.model_dump(exclude_none=False)
        incomplete_data = {
            "type_name": full_data["type_name"],
            "private_attribute_constructor": full_data["private_attribute_constructor"],
            "private_attribute_input_cache": full_data["private_attribute_input_cache"],
            "private_attribute_id": full_data["private_attribute_id"],
        }
        data_parsed = parse_model_dict(incomplete_data, globals())
        compare_objects_from_dict(full_data, data_parsed, BETDisk)
