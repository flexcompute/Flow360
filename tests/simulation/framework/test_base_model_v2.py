import json
import os
import tempfile
from typing import Optional

import pydantic as pd
import pytest
import yaml

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import (
    Conflicts,
    Flow360BaseModel,
)
from flow360.log import set_logging_level

set_logging_level("DEBUG")


class BaseModelTestModel(Flow360BaseModel):
    some_value: pd.StrictFloat = pd.Field()

    model_config = pd.ConfigDict(include_hash=True)

    def preprocess(self, **kwargs):
        self.some_value *= 2
        return super().preprocess(**kwargs)


class TempParams(Flow360BaseModel):
    some_value: pd.StrictFloat
    pseudo_field: BaseModelTestModel

    def preprocess(self, **kwargs):
        return super().preprocess(**kwargs)


class BaseModelWithConflictFields(Flow360BaseModel):
    some_value1: Optional[pd.StrictFloat] = pd.Field(None, alias="value1")
    some_value2: Optional[pd.StrictFloat] = pd.Field(None, alias="value2")

    model_config = pd.ConfigDict(
        conflicting_fields=[Conflicts(field1="some_value1", field2="some_value2")]
    )

    def preprocess(self, **kwargs):
        self.some_value1 *= 2
        return super().preprocess(self, **kwargs)


def test_help():
    Flow360BaseModel().help()
    Flow360BaseModel().help(methods=True)


def test_copy():
    base_model = BaseModelTestModel(some_value=123)
    base_model_copy = base_model.copy()
    assert base_model_copy.some_value == 123
    base_model.some_value = 12345
    assert base_model_copy.some_value == 123


def test_conflict():
    base_model = BaseModelWithConflictFields(some_value1=12.3)
    with pytest.raises(
        pd.ValidationError,
        match="some_value1 and some_value2 cannot be specified at the same time.",
    ):
        base_model.some_value2 = 12.3
    with pytest.raises(
        pd.ValidationError,
        match="some_value1 and some_value2 cannot be specified at the same time.",
    ):
        base_model.value2 = 12.3


def test_from_file():
    file_content = {"some_value": 321}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model = BaseModelTestModel.from_file(temp_file_name)
        assert base_model.some_value == 321
    finally:
        os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model = BaseModelTestModel.from_file(temp_file_name)
        assert base_model.some_value == 321
    finally:
        os.remove(temp_file_name)


def test_dict_from_file():
    file_content = {
        "some_value": 3210,
        "hash": "e6d346f112fc2ba998a286f9736ce389abb79f154dc84a104d3b4eb8ba4d4529",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model_dict = BaseModelTestModel._dict_from_file(temp_file_name)
        assert base_model_dict["some_value"] == 3210
    finally:
        os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model_dict = BaseModelTestModel._dict_from_file(temp_file_name)
        assert base_model_dict["some_value"] == 3210
    finally:
        os.remove(temp_file_name)


def test_to_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        base_model = BaseModelTestModel(some_value=1230)
        temp_file_name = temp_file.name

    try:
        base_model.to_file(temp_file_name)
        with open(temp_file_name) as fp:
            base_model_dict = json.load(fp)
            assert base_model_dict["some_value"] == 1230
            assert "hash" in base_model_dict
    finally:
        os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        base_model.to_file(temp_file_name)
        with open(temp_file_name) as fp:
            base_model_dict = yaml.load(fp, Loader=yaml.Loader)
            assert base_model_dict["some_value"] == 1230
    finally:
        os.remove(temp_file_name)


def test_preprocess():
    value = 123
    test_params = TempParams(pseudo_field=BaseModelTestModel(some_value=value), some_value=value)
    test_params = test_params.preprocess(params=test_params)
    assert test_params.some_value == value
    assert test_params.pseudo_field.some_value == value * 2
