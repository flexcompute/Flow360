import json
import os
import tempfile

import pydantic as pd
import pytest
import yaml

from flow360.component.flow360_params.flow360_params import Flow360BaseModel
from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.log import set_logging_level

set_logging_level("DEBUG")


class BaseModelTestModel(Flow360BaseModel):
    some_value: pd.StrictFloat = pd.Field()

    model_config = pd.ConfigDict(include_hash=True)


def test_help():
    Flow360BaseModel().help()
    Flow360BaseModel().help(methods=True)


def test_copy():
    base_model = BaseModelTestModel(some_value=123)
    base_model_copy = base_model.copy()
    assert base_model_copy.some_value == 123
    base_model.some_value = 12345
    assert base_model_copy.some_value == 123


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
        base_model_dict = BaseModelTestModel.dict_from_file(temp_file_name)
        assert base_model_dict["some_value"] == 3210
    finally:
        os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model_dict = BaseModelTestModel.dict_from_file(temp_file_name)
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


def test_from_json_yaml():
    file_content = {
        "some_value": 3210,
        "hash": "e6d346f112fc2ba998a286f9736ce389abb79f154dc84a104d3b4eb8ba4d4529",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model = BaseModelTestModel.from_json(temp_file_name)
        assert base_model.some_value == 3210
    finally:
        os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        base_model = BaseModelTestModel.from_yaml(temp_file_name)
        assert base_model.some_value == 3210
    finally:
        os.remove(temp_file_name)

    file_content = {"some_value": "43210", "hash": "aasdasd"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(file_content, temp_file)
        temp_file.flush()
        temp_file_name = temp_file.name

    print(json.dumps(file_content, indent=4))
    try:
        with pytest.raises(pd.ValidationError, match=r" Input should be a valid number"):
            base_model = BaseModelTestModel.from_json(temp_file_name)
    finally:
        os.remove(temp_file_name)


def test_add_type_field():
    ## Note: May need to be properly implemented.
    assert "_type" in BaseModelTestModel.model_fields


def test_generate_docstring():
    assert "some_value" in BaseModelTestModel.__doc__
