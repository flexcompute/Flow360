import json
import tempfile

import pydantic.v1 as pd
import pytest

import flow360.component.v1.units as u
import flow360.v1 as v1
from flow360.exceptions import Flow360NotImplementedError, Flow360RuntimeError
from flow360.v1 import Flow360Params

params_old_version = {
    "version": "0.2.0b01",
    "unitSystem": {"name": "SI"},
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
}

params_current_version = {
    "version": "24.11.0",
    "unitSystem": {"name": "SI"},
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "f097ce8e22c9a5f2b27b048aa775b74169fc13b2b60ab05c2913a8165d8c22c9",
}

params_no_version = {
    "unitSystem": {"name": "SI"},
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "momentCenter": {"units": "m", "value": [0, 0, 0]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"velocity": {"value": 286.0, "units": "m/s"}},
}

params_no_unit_system = {
    "version": "24.11.0",
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "momentCenter": {"units": "m", "value": [0, 0, 0]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "f097ce8e22c9a5f2b27b048aa775b74169fc13b2b60ab05c2913a8165d8c22c9",
}

params_no_hash = {
    "version": "24.11.0",
    "unitSystem": {"name": "SI"},
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "momentCenter": {"units": "m", "value": [0, 0, 0]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
}

params_wrong_hash = {
    "version": "24.11.0",
    "unitSystem": {"name": "SI"},
    "boundaries": {},
    "geometry": {
        "refArea": {"units": "m**2", "value": 1.15315084119231},
        "momentLength": {"units": "m", "value": [1.47602, 0.801672958512342, 1.47602]},
        "momentCenter": {"units": "m", "value": [0, 0, 0]},
        "meshUnit": {"units": "m", "value": 1.0},
    },
    "freestream": {"modelType": "FromVelocity", "velocity": {"value": 286.0, "units": "m/s"}},
    "hash": "invalid",
}


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_import_no_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    # when there is no unit system, we run the updater which should fail due to dimensioned fields
    with pytest.raises(pd.ValidationError):
        Flow360Params(temp_file.name)


def test_import_no_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_unit_system, temp_file)

    with pytest.raises(Flow360RuntimeError):
        with v1.flow360_unit_system:
            Flow360Params(temp_file.name)


def test_import_with_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    params = Flow360Params(temp_file.name)
    assert params
    assert params.unit_system == v1.SI_unit_system


def test_import_with_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    with pytest.raises(Flow360RuntimeError):
        with v1.SI_unit_system:
            Flow360Params(temp_file.name)


def test_copy_with_unit_system_no_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    # the unit system should be SI unit system imported from file
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the unit system gets copied from old file
    params_copy = params.copy()
    assert params_copy


def test_copy_with_unit_system_with_context():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_current_version, temp_file)

    # the unit system should be SI unit system imported from file
    params = Flow360Params(temp_file.name)
    assert params

    # passes, the models are consistent
    with v1.SI_unit_system:
        params_copy = params.copy()
        assert params_copy

    # fails, the models are inconsistent
    with v1.CGS_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy = params.copy()


def test_create_no_unit_system_no_context():
    with pytest.raises(Flow360RuntimeError):
        v1.Flow360Params(
            geometry=v1.Geometry(
                ref_area=1 * u.m**2,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=v1.air,
            freestream=v1.FreestreamFromVelocity(velocity=286 * u.m / u.s),
            time_stepping=v1.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=v1.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
        )


def test_create_with_unit_system_no_context():
    with pytest.raises(Flow360RuntimeError):
        v1.Flow360Params(
            geometry=v1.Geometry(
                ref_area=1 * u.m**2,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=v1.air,
            freestream=v1.FreestreamFromVelocity(velocity=286 * u.m / u.s),
            time_stepping=v1.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=v1.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
            unit_system=v1.SI_unit_system,
        )


def test_create_no_unit_system_with_context():
    with v1.SI_unit_system:
        v1.Flow360Params(
            geometry=v1.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            boundaries={},
            fluid_properties=v1.air,
            freestream=v1.FreestreamFromVelocity(velocity=286),
            time_stepping=v1.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=v1.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
        )


def test_create_with_unit_system_with_context():
    with v1.SI_unit_system:
        v1.Flow360Params(
            geometry=v1.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            boundaries={},
            fluid_properties=v1.air,
            freestream=v1.FreestreamFromVelocity(velocity=286),
            time_stepping=v1.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=v1.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
            unit_system=v1.SI_unit_system,
        )

    with pytest.raises(Flow360RuntimeError):
        with v1.CGS_unit_system:
            v1.Flow360Params(
                geometry=v1.Geometry(
                    ref_area=1,
                    moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                    moment_center=(1, 2, 3) * u.flow360_length_unit,
                    mesh_unit=u.mm,
                ),
                boundaries={},
                fluid_properties=v1.air,
                freestream=v1.FreestreamFromVelocity(velocity=286),
                time_stepping=v1.UnsteadyTimeStepping(
                    max_pseudo_steps=500,
                    CFL=v1.AdaptiveCFL(),
                    time_step_size=1.2 * u.s,
                    physical_steps=10,
                ),
                unit_system=v1.SI_unit_system,
            )


def test_change_version():
    with v1.SI_unit_system:
        params = Flow360Params(**params_no_hash)
    with pytest.raises(ValueError):
        params.version = "changed"


def test_parse_with_version():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_hash, temp_file)

    params = Flow360Params(temp_file.name)
    assert params.version == v1.__version__


def test_parse_no_version():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_no_version, temp_file)

    with pytest.raises(pd.ValidationError):
        Flow360Params(temp_file.name)


def test_parse_wrong_version():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_old_version, temp_file)

    with pytest.raises(Flow360NotImplementedError):
        Flow360Params(temp_file.name)


def test_parse_with_hash():
    with pytest.raises(pd.ValidationError):
        with v1.SI_unit_system:
            Flow360Params(**params_current_version)


def test_parse_no_hash():
    with v1.SI_unit_system:
        params = Flow360Params(**params_no_hash)
    assert params_no_hash.get("hash") is None
    assert not hasattr(params, "hash")


def test_parse_wrong_hash():
    with pytest.raises(pd.ValidationError):
        with v1.SI_unit_system:
            Flow360Params(**params_wrong_hash)
