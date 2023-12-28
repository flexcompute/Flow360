"""
Module exposing utilities for the validation service
"""
import json
import tempfile

import pydantic as pd

from .component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    NavierStokesSolver,
    SpalartAllmaras,
)
from .component.flow360_params.unit_system import UnitSystem, unit_system_manager
from .exceptions import Flow360ConfigurationError


def check_unit_system(unit_system):
    if not isinstance(unit_system, UnitSystem):
        raise ValueError(f"Incorrect unit system provided {unit_system=}, expected type UnitSystem")

    if unit_system_manager.current is not None:
        raise RuntimeError(
            f"Services cannot be used inside unit system context: {unit_system_manager.current.system_repr()}."
        )


def get_default_params(unit_system_context):
    """
    example of generating default case settings.
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    """

    check_unit_system(unit_system_context)

    with unit_system_context:
        params = Flow360Params(
            geometry=Geometry(
                ref_area=1, moment_center=(0, 0, 0), moment_length=(1, 1, 1), mesh_unit=1
            ),
            boundaries={},
            freestream=FreestreamFromVelocity.construct(),
            navier_stokes_solver=NavierStokesSolver(),
            turbulence_model_solver=SpalartAllmaras(),
        )

    return params


def get_default_retry(params_as_dict):
    """
    Return a default case file for a retry request
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_as_dict, temp_file)

    params = Flow360Params(temp_file.name)
    return params


def get_default_fork(params_as_dict):
    """
    Return a default case file for a fork request
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(params_as_dict, temp_file)

    params = Flow360Params(temp_file.name)
    return params


def validate_flow360_params_model(params_as_dict, unit_system_context):
    """
    Validate a params dict against the pydantic model
    """

    check_unit_system(unit_system_context)

    params_as_dict["unitSystem"] = unit_system_context.dict()
    values, fields_set, validation_errors = pd.validate_model(Flow360Params, params_as_dict)
    print(f"{values=}")
    print(f"{fields_set=}")

    # when validating freestream, errors from all Union options
    # will be returned. Need to reduce number of validation errors:
    # example when provided temperature -1
    # validation_errors=ValidationError(model='Flow360Params', errors=[
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0',
    # 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}},
    # {'loc': ('freestream', 'Reynolds'), 'msg': 'field required', 'type': 'value_error.missing'},
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0',
    # 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}},
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'},
    # {'loc': ('freestream', 'velocity'), 'msg': 'field required', 'type': 'value_error.missing'},
    # {'loc': ('freestream', 'Mach'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'},
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'},
    # {'loc': ('freestream', 'temperature'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'},
    # {'loc': ('freestream', 'Mach'), 'msg': 'unexpected value; permitted: 0',
    # 'type': 'value_error.const', 'ctx': {'given': 0.5, 'permitted': (0,)}},
    # {'loc': ('freestream', 'Mach_ref'), 'msg': 'field required', 'type': 'value_error.missing'},
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0',
    # 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}},
    # {'loc': ('freestream', 'velocity_ref'), 'msg': 'field required',
    # 'type': 'value_error.missing'}, {'loc': ('freestream', 'Mach'),
    # 'msg': 'extra fields not permitted', 'type': 'value_error.extra'},
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted',
    # 'type': 'value_error.extra'}, {'loc': ('freestream', 'temperature'),
    # 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}])

    # Gather dependency errors stemming from solver conversion if no validation errors exist
    if validation_errors is None:
        try:
            with unit_system_context:
                params = Flow360Params.parse_obj(params_as_dict)
            params.to_solver()
        except Flow360ConfigurationError as exc:
            validation_errors = [
                {"loc": exc.field, "msg": exc.msg, "type": "configuration_error"},
                {"loc": exc.dependency, "msg": exc.msg, "type": "configuration_error"},
            ]
    else:
        validation_errors = validation_errors.errors()

    print(f"{validation_errors=}")

    validation_warnings = None

    if validation_errors is not None:
        return validation_errors, validation_warnings

    return None, validation_warnings
