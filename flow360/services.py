"""
Module exposing utilities for the validation service
"""

import pydantic as pd

from .component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    SpalartAllmaras,
)
from .component.flow360_params.solvers import NavierStokesSolver
from .exceptions import Flow360ConfigurationError


def get_default_params(unit_system_context):
    """
    example of generating default case settings.
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    """

    with unit_system_context:
        params = Flow360Params(
            geometry=Geometry(
                ref_area=1, moment_center=(0, 0, 0), moment_length=(1, 1, 1), mesh_unit=1
            ),
            freestream=FreestreamFromVelocity.construct(),
            navier_stokes_solver=NavierStokesSolver(),
            turbulence_model_solver=SpalartAllmaras(),
        )

    return params


def get_default_retry(params_as_dict):
    """
    Return a default case file for a retry request
    """

    params = Flow360Params(legacy_fallback=True, **params_as_dict)
    return params


def get_default_fork(params_as_dict):
    """
    Return a default case file for a fork request
    """

    params = Flow360Params(legacy_fallback=True, **params_as_dict)
    return params


def validate_flow360_params_model(params_as_dict):
    """
    Validate a params dict against the pydantic model
    """
    values, fields_set, validation_errors = pd.validate_model(Flow360Params, params_as_dict)
    print(f"{values=}")
    print(f"{fields_set=}")
    print(f"{validation_errors=}")

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
            params = Flow360Params.parse_obj(params_as_dict)
            params.to_solver()
        except Flow360ConfigurationError as exc:
            validation_errors = [
                {"loc": exc.field, "msg": exc.msg, "type": "configuration_error"},
                {"loc": exc.dependency, "msg": exc.msg, "type": "configuration_error"},
            ]
    else:
        validation_errors = validation_errors.errors()

    # Check if all validation loc paths are valid params dict paths that can be traversed
    for error in validation_errors:
        current = params_as_dict
        for field in error["loc"][:-1]:
            if current.get(field):
                current = current.get(field)
            else:
                errors_as_list = list(error["loc"])
                errors_as_list.remove(field)
                error["loc"] = tuple(errors_as_list)

    validation_warnings = []

    if validation_errors is not None:
        return validation_errors, validation_warnings

    return None, validation_warnings
