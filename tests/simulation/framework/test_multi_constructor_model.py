import json
import os
import tempfile
from typing import Optional

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.cached_model_base import (
    _MultiConstructorModelBase,
    parse_model_dict,
)
from flow360.component.simulation.operating_condition import (
    AerospaceCondition,
    ThermalState,
    AerospaceConditionCache,
)


@pytest.fixture
def get_aerospace_condition_default():
    return AerospaceCondition(
        velocity_magnitude=0.8 * u.km / u.s,
        alpha=5 * u.deg,
        thermal_state=ThermalState(),
        reference_velocity_magnitude=100 * u.m / u.s,
    )


@pytest.fixture
def get_aerospace_condition_using_from():
    return AerospaceCondition.from_mach(
        mach=0.8,
        alpha=5 * u.deg,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000 * u.m),
    )


def test_full_model(get_aerospace_condition_default, get_aerospace_condition_using_from):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=True)
    data_parsed = parse_model_dict(full_data, globals())
    assert sorted(data_parsed.items()) == sorted(full_data.items())

    full_data = get_aerospace_condition_using_from.model_dump(exclude_none=True)
    data_parsed = parse_model_dict(full_data, globals())
    assert sorted(data_parsed.items()) == sorted(full_data.items())


def test_incomplete_model(get_aerospace_condition_default, get_aerospace_condition_using_from):
    full_data = get_aerospace_condition_default.model_dump(exclude_none=True)
    incomplete_data = full_data
    full_data["private_attribute_input_cache"] = None
    data_parsed = parse_model_dict(incomplete_data, globals())
    assert sorted(data_parsed.items()) == sorted(full_data.items())

    full_data = get_aerospace_condition_using_from.model_dump(exclude_none=True)
    incomplete_data = {
        "type_name": full_data["type_name"],
        "private_attribute_constructor": full_data["private_attribute_constructor"],
        "private_attribute_input_cache": full_data["private_attribute_input_cache"],
    }
    data_parsed = parse_model_dict(incomplete_data, globals())
    assert sorted(data_parsed.items()) == sorted(full_data.items())

