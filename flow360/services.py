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
from .component.flow360_params.params_base import flow360_json_encoder
from .component.flow360_params.unit_system import (
    UnitSystem,
    unit_system_manager,
    SI_unit_system,
    CGS_unit_system,
    imperial_unit_system,
    flow360_unit_system,
)
from .exceptions import Flow360ConfigurationError


unit_system_map = {
    "SI": SI_unit_system,
    "CGS": CGS_unit_system,
    "Imperial": imperial_unit_system,
    "Flow360": flow360_unit_system,
}


def params_to_dict(params: Flow360Params) -> dict:
    params_as_dict = json.loads(params.json())

    if params.bet_disks is not None:
        params_as_dict["BETDisks"] = [
            json.loads(bet_disk.json(encoder=flow360_json_encoder)) for bet_disk in params.bet_disks
        ]

    return params_as_dict


def init_unit_system(unit_system_name):
    unit_system = unit_system_map.get(unit_system_name, None)
    if not isinstance(unit_system, UnitSystem):
        raise ValueError(f"Incorrect unit system provided {unit_system=}, expected type UnitSystem")

    if unit_system_manager.current is not None:
        raise RuntimeError(
            f"Services cannot be used inside unit system context. Used: {unit_system_manager.current.system_repr()}."
        )
    return unit_system


def remove_properties_with_prefix(data, prefix):
    if isinstance(data, dict):
        return {
            key: remove_properties_with_prefix(value, prefix)
            for key, value in data.items()
            if not key.startswith(prefix)
        }
    elif isinstance(data, list):
        return [remove_properties_with_prefix(item, prefix) for item in data]
    else:
        return data


def remove_dimensioned_type_none_leaves(data):
    if isinstance(data, dict):
        return {
            key: remove_dimensioned_type_none_leaves(value)
            for key, value in data.items()
            if not (
                isinstance(value, dict)
                and "value" in value
                and "units" in value
                and value["value"] is None
            )
        }
    elif isinstance(data, list):
        return [remove_dimensioned_type_none_leaves(item) for item in data if item is not None]
    else:
        return data


def get_default_params(unit_system_name):
    """
    example of generating default case settings.
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    """

    unit_system = init_unit_system(unit_system_name)

    with unit_system:
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


def validate_flow360_params_model(params_as_dict, unit_system_name):
    """
    Validate a params dict against the pydantic model
    """

    unit_system = init_unit_system(unit_system_name)

    # removing _add properties as these are only used in WebUI
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_add")
    params_as_dict = remove_dimensioned_type_none_leaves(params_as_dict)

    params_as_dict["unitSystem"] = unit_system.dict()
    values, fields_set, validation_errors = pd.validate_model(Flow360Params, params_as_dict)
    print(f"{values=}")
    print(f"{fields_set=}")

    # Gather dependency errors stemming from solver conversion if no validation errors exist
    if validation_errors is None:
        try:
            with unit_system:
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

    # Check if all validation loc paths are valid params dict paths that can be traversed
    if validation_errors is not None:
        for error in validation_errors:
            current = params_as_dict
            for field in error["loc"][:-1]:
                if current.get(field):
                    current = current.get(field)
                else:
                    errors_as_list = list(error["loc"])
                    errors_as_list.remove(field)
                    error["loc"] = tuple(errors_as_list)

        return validation_errors, validation_warnings

    return None, validation_warnings


def handle_case_submit(params_as_dict, unit_system_name):
    unit_system = init_unit_system(unit_system_name)
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_add")
    params_as_dict = remove_dimensioned_type_none_leaves(params_as_dict)

    with unit_system:
        params = Flow360Params(**params_as_dict)

    solver_json = params.to_flow360_json()
    solver_dict = json.loads(solver_json)

    return params, solver_dict
