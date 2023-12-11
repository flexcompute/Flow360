import pytest
import pydantic as pd

import json
import tempfile

import flow360
from flow360 import Flow360Params
from flow360.exceptions import RuntimeError

 

params_old_version = {
    "version": "0.2.0b01",
    "geometry": {
        "refArea": {
            "units": "m**2",
            "value": 1.15315084119231
        },
        "momentLength": {
            "units": "m",
            "value": [
                1.47602,
                0.801672958512342,
                1.47602
            ]
        },
        "meshUnit": {
            "units": "m",
            "value": 1.0
        }
    },
    "freestream": {
        "velocity": {
            "value": 286.0,
            "units": "m/s"
        }
    }
}

params_current_version = {
    "version": "0.2.0b16",
    "geometry": {
        "refArea": {
            "units": "m**2",
            "value": 1.15315084119231
        },
        "momentLength": {
            "units": "m",
            "value": [
                1.47602,
                0.801672958512342,
                1.47602
            ]
        },
        "meshUnit": {
            "units": "m",
            "value": 1.0
        }
    },
    "freestream": {
        "velocity": {
            "value": 286.0,
            "units": "m/s"
        }
    },
    "hash": "f097ce8e22c9a5f2b27b048aa775b74169fc13b2b60ab05c2913a8165d8c22c9"
}


params_no_version = {
    "geometry": {
        "refArea": {
            "units": "m**2",
            "value": 1.15315084119231
        },
        "momentLength": {
            "units": "m",
            "value": [
                1.47602,
                0.801672958512342,
                1.47602
            ]
        },
        "meshUnit": {
            "units": "m",
            "value": 1.0
        }
    },
    "freestream": {
        "velocity": {
            "value": 286.0,
            "units": "m/s"
        }
    }
}





@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_import_no_unit_system_no_context():


    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    params = Flow360Params(temp_file.name)
    assert params

    # the unit system should be flow360 unit system if imported from file and no unit system loaded


def test_import_no_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    with flow360.flow360_unit_system:
        with pytest.raises(RuntimeError):
            params = Flow360Params(temp_file.name)



def test_import_with_unit_system_no_context():

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    params = Flow360Params(temp_file.name)
    assert params
    assert params.unit_system == 'SI'


def test_import_with_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    with flow360.flow360_unit_system:
        with pytest.raises(RuntimeError):
            params = Flow360Params(temp_file.name)


def test_copy_no_unit_system_no_context():
    pass



def test_copy_with_unit_system_no_context():
    pass



def test_copy_no_unit_system_with_context():
    pass



def test_copy_with_unit_system_with_context():
    pass



def test_create_no_unit_system_no_context():
    pass



def test_create_with_unit_system_no_context():
    pass



def test_create_no_unit_system_with_context():
    pass



def test_create_with_unit_system_with_context():
    pass



def test_change_version():
    params = Flow360Params(**params_current_version)
    with pytest.raises(pd.ValidationError):
        params.version = 'changed'


def test_parse_with_version():
    params = Flow360Params(**params_current_version)
    assert params.version == flow360.__version__


def test_parse_no_version():
    params = Flow360Params(**params_no_version)
    assert params.version == flow360.__version__




# def test_parse_wrong_version():
#     print(flow360.__version__)
#     params = Flow360Params(**params_old_version)
#     assert params.version == flow360.__version__


def test_parse_with_hash():
    pass



def test_parse_no_hash():
    pass



def test_parse_wrong_hash():
    pass



