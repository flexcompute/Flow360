import json
import tempfile

import pydantic as pd
import pytest

import flow360
import flow360.units as u
from flow360 import Flow360Params
from flow360.exceptions import Flow360RuntimeError

params_old_version = {
    "version": "0.2.0b01",
    "unitSystem": {"name": "SI"},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
}

params_current_version = {
    "version": "0.2.0b16",
    "unitSystem": {"name": "SI"},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "f097ce8e22c9a5f2b27b048aa775b74169fc13b2b60ab05c2913a8165d8c22c9",
}

params_no_version = {
    "unitSystem": {"name": "SI"},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
}

params_no_unit_system = {
    "version": "0.2.0b16",
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "f097ce8e22c9a5f2b27b048aa775b74169fc13b2b60ab05c2913a8165d8c22c9",
}

params_no_hash = {
    "version": "0.2.0b16",
    "unitSystem": {"name": "SI"},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
}

params_wrong_hash = {
    "version": "0.2.0b16",
    "unitSystem": {"name": "SI"},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "invalid",
}


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_import_no_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    # the unit system should be flow360 unit system if imported from file and no unit system loaded
    params = Flow360Params(temp_file.name)
    assert params


def test_import_no_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    with pytest.raises(Flow360RuntimeError):
        with flow360.flow360_unit_system:
            params = Flow360Params(temp_file.name)


def test_import_with_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    params = Flow360Params(temp_file.name)
    assert params
    assert params.unit_system == flow360.SI_unit_system


def test_import_with_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    with pytest.raises(Flow360RuntimeError):
        with flow360.SI_unit_system:
            params = Flow360Params(temp_file.name)


def test_copy_no_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    # the unit system should be flow360 unit system if imported from file and no unit system loaded
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the unit system gets copied from old file
    params_copy = params.copy()
    assert params_copy


def test_copy_with_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    # the unit system should be SI unit system imported from file
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the unit system gets copied from old file
    params_copy = params.copy()
    assert params_copy


def test_copy_no_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    # the unit system should be flow360 unit system if imported from file and no unit system loaded
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the models are consistent
    with flow360.flow360_unit_system:
        params_copy = params.copy()
        assert params_copy

    # fails, the models are inconsistent
    with flow360.SI_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy = params.copy()


def test_copy_with_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    # the unit system should be SI unit system imported from file
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the models are consistent
    with flow360.SI_unit_system:
        params_copy = params.copy()
        assert params_copy

    # fails, the models are inconsistent
    with flow360.CGS_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy = params.copy()


def test_create_no_unit_system_no_context():
    with pytest.raises(Flow360RuntimeError):
        params = flow360.Flow360Params(
            geometry=flow360.Geometry(
                ref_area=1 * u.m**2,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=flow360.air,
            freestream=flow360.FreestreamFromVelocity(velocity=286 * u.m / u.s),
            time_stepping=flow360.TimeStepping(
                max_pseudo_steps=500, CFL=flow360.AdaptiveCFL(), time_step_size=1.2 * u.s
            ),
        )


def test_create_with_unit_system_no_context():
    with pytest.raises(Flow360RuntimeError):
        params = flow360.Flow360Params(
            geometry=flow360.Geometry(
                ref_area=1 * u.m**2,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=flow360.air,
            freestream=flow360.FreestreamFromVelocity(velocity=286 * u.m / u.s),
            time_stepping=flow360.TimeStepping(
                max_pseudo_steps=500, CFL=flow360.AdaptiveCFL(), time_step_size=1.2 * u.s
            ),
            unit_system=flow360.SI_unit_system,
        )


def test_create_no_unit_system_with_context():
    with flow360.SI_unit_system:
        params = flow360.Flow360Params(
            geometry=flow360.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=flow360.air,
            freestream=flow360.FreestreamFromVelocity(velocity=286),
            time_stepping=flow360.TimeStepping(
                max_pseudo_steps=500, CFL=flow360.AdaptiveCFL(), time_step_size=1.2 * u.s
            ),
        )


def test_create_with_unit_system_with_context():
    with flow360.SI_unit_system:
        params = flow360.Flow360Params(
            geometry=flow360.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=flow360.air,
            freestream=flow360.FreestreamFromVelocity(velocity=286),
            time_stepping=flow360.TimeStepping(
                max_pseudo_steps=500, CFL=flow360.AdaptiveCFL(), time_step_size=1.2 * u.s
            ),
            unit_system=flow360.SI_unit_system,
        )

    with pytest.raises(Flow360RuntimeError):
        with flow360.CGS_unit_system:
            params = flow360.Flow360Params(
                geometry=flow360.Geometry(
                    ref_area=1,
                    moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                    moment_center=(1, 2, 3) * u.flow360_length_unit,
                    mesh_unit=u.mm,
                ),
                fluid_properties=flow360.air,
                freestream=flow360.FreestreamFromVelocity(velocity=286),
                time_stepping=flow360.TimeStepping(
                    max_pseudo_steps=500, CFL=flow360.AdaptiveCFL(), time_step_size=1.2 * u.s
                ),
                unit_system=flow360.SI_unit_system,
            )


def test_change_version():
    with flow360.SI_unit_system:
        params = Flow360Params(**params_no_hash)
    with pytest.raises(ValueError):
        params.version = "changed"


def test_parse_with_version():
    with flow360.SI_unit_system:
        params = Flow360Params(**params_no_hash)
    assert params.version == flow360.__version__


def test_parse_no_version():
    with flow360.SI_unit_system:
        params = Flow360Params(**params_no_version)
    assert params.version == flow360.__version__


# def test_parse_wrong_version():
#     print(flow360.__version__)
#     params = Flow360Params(**params_old_version)
#     assert params.version == flow360.__version__


def test_parse_with_hash():
    with pytest.raises(pd.ValidationError):
        with flow360.SI_unit_system:
            params = Flow360Params(**params_current_version)


def test_parse_no_hash():
    with flow360.SI_unit_system:
        params = Flow360Params(**params_no_hash)
    assert params_no_hash.get("hash") is None
    assert not hasattr(params, "hash")


def test_parse_wrong_hash():
    with pytest.raises(pd.ValidationError):
        with flow360.SI_unit_system:
            params = Flow360Params(**params_wrong_hash)
