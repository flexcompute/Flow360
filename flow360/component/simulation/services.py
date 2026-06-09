"""Simulation services module."""

import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pydantic as pd

# pylint: disable=unused-import  # relay exports consumed by compute pipeline scripts and tests
from flow360_schema.framework.expression.registry import clear_context
from flow360_schema.framework.physical_dimensions import Angle, Length
from flow360_schema.models.entities.volume_entities import Box
from flow360_schema.models.simulation.models.bet.bet_translator_interface import (
    generate_polar_file_name_list,
    translate_xfoil_c81_to_bet_dict,
    translate_xrotor_dfdc_to_bet_dict,
)
from flow360_schema.models.simulation.models.volume_models import BETDisk
from flow360_schema.models.simulation.operating_condition.operating_condition import (
    GenericReferenceCondition,
    ThermalState,
)
from flow360_schema.models.simulation.services import (
    ValidationCalledBy,
    _determine_validation_level,
    _get_default_reference_geometry,
    _parse_root_item_type_from_simulation_json,
    apply_simulation_setting_to_entity_info,
    get_default_params,
    merge_geometry_entity_info,
    update_simulation_json,
)
from flow360_schema.models.simulation.services import (
    validate_model as _schema_validate_model,
)
from flow360_schema.models.simulation.simulation_params import SimulationParams
from flow360_schema.models.simulation.validation.validation_context import ALL
from flow360_schema.models.simulation.validation.validation_service import (
    initialize_variable_space,
)

# pylint: enable=unused-import
from pydantic import TypeAdapter

from flow360.component.simulation.unit_system import _dimensioned_type_serializer, u
from flow360.exceptions import Flow360ValueError


def validate_model(  # pylint: disable=too-many-locals
    *,
    params_as_dict,
    validated_by: ValidationCalledBy,
    root_item_type: Union[Literal["Geometry", "SurfaceMesh", "VolumeMesh"], None],
    validation_level: Union[
        Literal["SurfaceMesh", "VolumeMesh", "Case", "All"], list, None
    ] = ALL,  # Fix implicit string concatenation
) -> Tuple[Optional[SimulationParams], Optional[list], List[Dict[str, Any]]]:
    """
    Validate a params dict against the pydantic model.

    Parameters
    ----------
    params_as_dict : dict
        The parameters dictionary to validate.
    validated_by : ValidationCalledBy
        Indicator of where the `validate_model` function is called. Allowing generation of helpful messages.
    root_item_type : Union[Literal["Geometry", "SurfaceMesh", "VolumeMesh"], None],
        The root item type for validation. If None then no context-aware validation is performed.
    validation_level : Literal["SurfaceMesh", "VolumeMesh", "Case", "All"] or a list of literals, optional
        The validation level, default is ALL. Also a list can be provided, eg: ["SurfaceMesh", "VolumeMesh"]

    Returns
    -------
    validated_param : SimulationParams or None
        The validated parameters if successful, otherwise None.
    validation_errors : list or None
        A list of validation errors if any occurred.
    validation_warnings : list
        A list of validation warnings (empty list if no warnings were recorded).
    """
    return _schema_validate_model(
        params_as_dict=params_as_dict,
        validated_by=validated_by,
        root_item_type=root_item_type,
        validation_level=validation_level,
    )


def _serialize_unit_in_dict(data):
    """
    Recursively serialize unit type data in a dictionary or list.

    For unyt_quantity objects, converts them to {"value": item.value, "units": item.units.expr}
    Handles nested dictionaries, lists, and other basic types.

    Parameters:
    -----------
    data : any
        The data to serialize, can be a dictionary, list, unyt_quantity or other basic types

    Returns:
    --------
    any
        The serialized data with unyt_quantity objects converted to dictionaries
    """

    if isinstance(data, (u.unyt_quantity, u.unyt_array)):
        return _dimensioned_type_serializer(data)

    if isinstance(data, dict):
        return {key: _serialize_unit_in_dict(value) for key, value in data.items()}

    if isinstance(data, list):
        return [_serialize_unit_in_dict(item) for item in data]

    return data


_angle_adapter = TypeAdapter(Angle.Float64)
_length_adapter = TypeAdapter(Length.Float64)

_UNIT_TYPE_ADAPTERS = {
    "angle": _angle_adapter,
    "length": _length_adapter,
}


def _validate_unit_string(unit_str: str, unit_kind: Literal["angle", "length"]):
    """
    Validate the unit string from request against the specified unit kind.

    Parameters
    ----------
    unit_str : str
        JSON-encoded or plain unit string.
    unit_kind : str
        One of "angle" or "length".
    """
    adapter = _UNIT_TYPE_ADAPTERS[unit_kind]
    try:
        unit_dict = json.loads(unit_str)
        return adapter.validate_python(unit_dict)
    except json.JSONDecodeError:
        return adapter.validate_python(u.Unit(unit_str))


def translate_dfdc_xrotor_bet_disk(
    *,
    geometry_file_content: str,
    length_unit: str,
    angle_unit: str,
    file_format: str,
) -> list[dict]:
    """
    Run the BET Disk translator for an XROTOR or DFDC input file.
    Returns the dict of BETDisk.
    """
    # pylint: disable=no-member
    errors = []
    bet_dict_list = []
    try:
        length_unit = _validate_unit_string(length_unit, "length")
        angle_unit = _validate_unit_string(angle_unit, "angle")
        bet_disk_dict = translate_xrotor_dfdc_to_bet_dict(
            geometry_file_content=geometry_file_content,
            length_unit=length_unit,
            angle_unit=angle_unit,
            file_format=file_format,
        )
        bet_dict_list.append(_serialize_unit_in_dict(bet_disk_dict))
    except (pd.ValidationError, Flow360ValueError, ValueError) as e:
        # Expected exceptions
        errors.append(str(e))
    return bet_dict_list, errors


def translate_xfoil_c81_bet_disk(
    *,
    geometry_file_content: str,
    polar_file_contents_dict: dict,
    length_unit: str,
    angle_unit: str,
    file_format: str,
) -> list[dict]:
    """
    Run the BET Disk translator for an XFOIL or C81 input file.
    Returns the dict of BETDisk.
    """
    # pylint: disable=no-member
    errors = []
    bet_dict_list = []
    try:
        length_unit = _validate_unit_string(length_unit, "length")
        angle_unit = _validate_unit_string(angle_unit, "angle")
        polar_file_name_list = generate_polar_file_name_list(
            geometry_file_content=geometry_file_content
        )
        polar_file_contents_list = []
        polar_file_extensions = []
        for file_name_list in polar_file_name_list:
            file_contents_list = []
            for file_name in file_name_list:
                if file_name not in polar_file_contents_dict.keys():
                    raise ValueError(
                        f"The {file_format} polar file: {file_name} is missing. Please check the uploaded polar files."
                    )
                file_contents_list.append(polar_file_contents_dict.get(file_name))
            polar_file_contents_list.append(file_contents_list)
            polar_file_extensions.append(os.path.splitext(file_name_list[0])[1])
        bet_disk_dict = translate_xfoil_c81_to_bet_dict(
            geometry_file_content=geometry_file_content,
            polar_file_contents_list=polar_file_contents_list,
            polar_file_extensions=polar_file_extensions,
            length_unit=length_unit,
            angle_unit=angle_unit,
            file_format=file_format,
        )
        bet_dict_list.append(_serialize_unit_in_dict(bet_disk_dict))
    except (pd.ValidationError, Flow360ValueError, ValueError) as e:
        # Expected exceptions
        errors.append(str(e))
    return bet_dict_list, errors
