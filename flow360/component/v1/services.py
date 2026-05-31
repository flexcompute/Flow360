"""
Module exposing utilities for the validation service
"""

import json
from typing import Union

import pydantic.v1 as pd

from flow360.component.utils import remove_properties_with_prefix
from flow360.component.v1.flow360_params import (
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
)
from flow360.component.v1.params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _schema_optional_toggle_name,
    flow360_json_encoder,
)
from flow360.component.v1.solvers import NavierStokesSolver, SpalartAllmaras
from flow360.component.v1.unit_system import (
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    flow360_unit_system,
    imperial_unit_system,
    unit_system_manager,
)
from flow360.exceptions import Flow360ConfigurationError

unit_system_map = {
    "SI": SI_unit_system,
    "CGS": CGS_unit_system,
    "Imperial": imperial_unit_system,
    "Flow360": flow360_unit_system,
}


def _is_dimensioned_value_dict(value):
    return isinstance(value, dict) and "value" in value and "units" in value


def _add_nested_object_flag(params: Flow360BaseModel, params_as_dict: dict, level: int = 0) -> dict:
    if isinstance(params, Flow360BaseModel):
        for property_name, value in params.__dict__.items():
            # pylint: disable=protected-access
            alias_name = params._get_field_alias(property_name)
            key = property_name
            if alias_name is not None:
                key = alias_name

            if key in params_as_dict:
                item = params_as_dict[key]
                if (
                    not isinstance(params, Flow360SortableBaseModel)
                    and not _is_dimensioned_value_dict(item)
                    and isinstance(item, dict)
                    and level > 0
                ):
                    params_as_dict[_schema_optional_toggle_name(key)] = True
                params_as_dict[key] = _add_nested_object_flag(
                    value, params_as_dict[key], level=level + 1
                )
    elif isinstance(params, list):
        params_as_dict = [
            _add_nested_object_flag(item, params_as_dict[i], level=level + 1)
            for i, item in enumerate(params)
        ]

    return params_as_dict


def handle_add_nested_object_flag(params, params_as_dict: dict) -> dict:
    """will add _add<KeyName>=True flag to nested objects"""
    return _add_nested_object_flag(params, params_as_dict)


def params_to_dict(params: Flow360Params) -> dict:
    """
    converts Flow360Params to dictionary representation. For BET it removes all dimensioned fields as they are not
    supported yet by webUI
    """
    params_as_dict = json.loads(params.json())

    if params.bet_disks is not None:
        params_as_dict["BETDisks"] = [
            json.loads(bet_disk.json(encoder=flow360_json_encoder)) for bet_disk in params.bet_disks
        ]

    params_as_dict = handle_add_nested_object_flag(params, params_as_dict)

    return params_as_dict


def init_unit_system(unit_system_name) -> UnitSystem:
    """Returns UnitSystem object from string representation.

    Parameters
    ----------
    unit_system_name : ["SI", "CGS", "Imperial", "Flow360"]
        Unit system string representation

    Returns
    -------
    UnitSystem
        unit system

    Raises
    ------
    ValueError
        If unit system doesn't exist
    RuntimeError
        If this function is run inside unit system context
    """

    unit_system = unit_system_map.get(unit_system_name, None)
    if not isinstance(unit_system, UnitSystem):
        raise ValueError(
            f"Incorrect unit system provided for {unit_system_name} unit "
            f"system, got {unit_system=}, expected value of type UnitSystem"
        )

    if unit_system_manager.current is not None:
        raise RuntimeError(
            f"Services cannot be used inside unit system context. Used: {unit_system_manager.current.system_repr()}."
        )
    return unit_system


def remove_dimensioned_type_none_leaves(data):
    """
    Recursively removes leaves from a nested dictionary and its lists where the value
    is `None` and the structure contains keys 'value' and 'units' in the dictionary.

    Parameters
    ----------
    data : dict or list or scalar
        The input data, which can be a nested dictionary, a list, or a scalar value.

    Returns
    -------
    dict or list or scalar
        Processed data with leaves removed where 'value' is `None` in dictionaries.
    """

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
    if isinstance(data, list):
        return [remove_dimensioned_type_none_leaves(item) for item in data if item is not None]
    return data


def remove_empty_entries(data, exclude):
    """
    Recursively removes empty dictionaries apart from excluded keys

    Parameters
    ----------
    data : dict
        The input dictionary.

    exclude : list[str]
        List of excluded keys to keep when traversing the data

    Returns
    -------
    dict
        Processed data with empty dictionary entries removed.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    for key, value in list(data.items()):
        if isinstance(value, dict):
            if value:
                remove_empty_entries(value, exclude)
            elif key not in exclude:
                del data[key]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_empty_entries(item, exclude)

    return data


def get_default_params(unit_system_name):
    """
    Returns default parameters in a given unit system. The defaults are not correct Flow360Params object as they may
    contain empty required values. When generating default case settings:
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    Parameters
    ----------
    unit_system_name : str
        The name of the unit system to use for parameter initialization.

    Returns
    -------
    Flow360Params
        Default parameters for Flow360 simulation.

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
    Returns Flow360Params object for a retry request. It will perform update if neccessary.
    """

    remove_empty_entries(params_as_dict, exclude=["boundaries"])

    params = Flow360Params(legacy_fallback=True, **params_as_dict)
    return params


def get_default_fork(params_as_dict):
    """
    Returns Flow360Params object for a retry request. It will perform update if neccessary.
    """

    remove_empty_entries(params_as_dict, exclude=["boundaries"])

    params = Flow360Params(legacy_fallback=True, **params_as_dict)
    return params


def validate_model(params_as_dict, unit_system_name):
    """
    Validate a params dict against the pydantic model
    """

    unit_system = init_unit_system(unit_system_name)

    # removing _add and _temp properties as these are only used in WebUI
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_add")
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_temp")
    params_as_dict = remove_dimensioned_type_none_leaves(params_as_dict)

    params_as_dict["unitSystem"] = unit_system.dict()

    remove_empty_entries(params_as_dict, exclude=["boundaries"])

    values, fields_set, validation_errors = pd.validate_model(Flow360Params, params_as_dict)
    print(f"{values=}")
    print(f"{fields_set=}")

    print(f"{validation_errors=}")

    validation_warnings = None

    # Check if all validation loc paths are valid params dict paths that can be traversed
    if validation_errors is not None:
        validation_errors = validation_errors.errors()
        for error in validation_errors:
            current = params_as_dict
            for field in error["loc"][:-1]:
                if (
                    isinstance(field, int)
                    and isinstance(current, list)
                    and field in range(0, len(current))
                ):
                    current = current[field]
                elif isinstance(field, str) and isinstance(current, dict) and current.get(field):
                    current = current.get(field)
                else:
                    errors_as_list = list(error["loc"])
                    errors_as_list.remove(field)
                    error["loc"] = tuple(errors_as_list)
    else:
        # Gather dependency errors stemming from solver conversion if no validation errors exist
        try:
            with unit_system:
                params = Flow360Params.parse_obj(params_as_dict)
            params.to_solver()
        except Flow360ConfigurationError as exc:
            validation_errors = [
                {"loc": exc.field, "msg": exc.msg, "type": "configuration_error"},
                {"loc": exc.dependency, "msg": exc.msg, "type": "configuration_error"},
            ]

    return validation_errors, validation_warnings


def handle_case_submit(params_as_dict: Union[dict, list], unit_system_name: str):
    """
    Handles case submit. Performs pydantic validation, converts units to solver units, and exports JSON representation.

    Parameters
    ----------
    params_as_dict : dict | list | Any

    unit_system_name : str

    Returns
    -------
    Flow360Params
    """
    unit_system = init_unit_system(unit_system_name)
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_add")
    params_as_dict = remove_properties_with_prefix(params_as_dict, "_temp")
    params_as_dict = remove_dimensioned_type_none_leaves(params_as_dict)

    with unit_system:
        params = Flow360Params(**params_as_dict)

    return params
