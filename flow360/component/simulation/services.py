"""Simulation services module."""

import json
import os
from typing import Any, Collection, Dict, List, Literal, Optional, Tuple, Union

import pydantic as pd
from flow360_schema.framework.physical_dimensions import Angle, Length

# pylint: disable=unused-import  # relay exports consumed by compute pipeline scripts and tests
from flow360_schema.framework.expression.registry import clear_context
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
from flow360_schema.models.simulation.validation.validation_service import (
    initialize_variable_space,
)

# pylint: enable=unused-import
from pydantic import TypeAdapter

from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.models.bet.bet_translator_interface import (
    generate_polar_file_name_list,
    translate_xfoil_c81_to_bet_dict,
    translate_xrotor_dfdc_to_bet_dict,
)

# pylint: disable=unused-import # For parse_model_dict
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.operating_condition.operating_condition import (
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import Box

# pylint: enable=unused-import
from flow360.component.simulation.simulation_params import SimulationParams

# Required for correct global scope initialization
from flow360.component.simulation.translator.solver_translator import (
    get_columnar_data_processor_json,
    get_solver_json,
)
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import _dimensioned_type_serializer, u
from flow360.component.simulation.validation.validation_context import ALL
from flow360.exceptions import Flow360TranslationError, Flow360ValueError
from flow360.version import __version__


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
        version_to=__version__,
    )


# pylint: disable=too-many-arguments
def _translate_simulation_json(
    input_params: SimulationParams,
    mesh_unit,
    target_name: str = None,
    translation_func=None,
    **kwargs,
):
    """
    Get JSON for surface meshing from a given simulation JSON.

    """
    translated_dict = None
    if mesh_unit is None:
        raise ValueError("Mesh unit is required for translation.")
    if isinstance(input_params, SimulationParams) is False:
        raise ValueError(
            "input_params must be of type SimulationParams. Instead got: " + str(type(input_params))
        )

    try:
        translated_dict = translation_func(input_params, mesh_unit, **kwargs)
    except Flow360TranslationError as err:
        raise ValueError(str(err)) from err
    except Exception as err:  # translation itself is not supposed to raise any other exception
        raise ValueError(
            f"Unexpected error translating to {target_name} json: " + str(err)
        ) from err

    if translated_dict == {}:
        raise ValueError(f"No {target_name} parameters found in given SimulationParams.")

    # pylint: disable=protected-access
    hash_value = SimulationParams._calculate_hash(translated_dict)
    return translated_dict, hash_value


def simulation_to_surface_meshing_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for surface meshing from a given simulation JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "surface meshing",
        get_surface_meshing_json,
    )


def simulation_to_volume_meshing_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for volume meshing from a given simulation JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "volume meshing",
        get_volume_meshing_json,
    )


def simulation_to_case_json(
    input_params: SimulationParams, mesh_unit, *, skip_selector_expansion: bool = False
):
    """Get JSON for case from a given simulation JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "case",
        get_solver_json,
        skip_selector_expansion=skip_selector_expansion,
    )


def simulation_to_columnar_data_processor_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for case postprocessing from a given simulation JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "case postprocessing",
        get_columnar_data_processor_json,
    )


def _get_mesh_unit(params_as_dict: dict) -> str:
    if params_as_dict.get("private_attribute_asset_cache") is None:
        raise ValueError("[Internal] failed to acquire length unit from simulation settings.")
    mesh_unit = params_as_dict["private_attribute_asset_cache"].get("project_length_unit")
    if mesh_unit is None:
        raise ValueError("[Internal] failed to acquire length unit from simulation settings.")
    return mesh_unit


def _process_surface_mesh(
    params: dict, root_item_type: str, mesh_unit: str
) -> Optional[Dict[str, Any]]:
    if root_item_type == "Geometry":
        sm_data, sm_hash_value = simulation_to_surface_meshing_json(params, mesh_unit)
        return {"data": json.dumps(sm_data), "hash": sm_hash_value}
    return None


def _process_volume_mesh(
    params: dict, root_item_type: str, mesh_unit: str, up_to: str
) -> Optional[Dict[str, Any]]:
    if up_to != "SurfaceMesh" and root_item_type != "VolumeMesh":
        vm_data, vm_hash_value = simulation_to_volume_meshing_json(params, mesh_unit)
        return {"data": json.dumps(vm_data), "hash": vm_hash_value}
    return None


def _process_case(params: dict, mesh_unit: str, up_to: str) -> Optional[Dict[str, Any]]:
    if up_to == "Case":
        case_data, case_hash_value = simulation_to_case_json(params, mesh_unit)
        return {"data": json.dumps(case_data), "hash": case_hash_value}
    return None


def generate_process_json(
    *,
    simulation_json: str,
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"],
    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"],
):
    """
    Generates process JSON based on the simulation parameters.

    This function processes the simulation parameters from a JSON string and generates the
    corresponding process JSON for SurfaceMesh, VolumeMesh, and Case based on the input parameters.

    Parameters
    ----------
    simulation_json : str
        The JSON string containing simulation parameters.
    root_item_type : Literal["Geometry", "SurfaceMesh", "VolumeMesh"]
        The root item type for the simulation (e.g., "Geometry", "VolumeMesh").
    up_to : Literal["SurfaceMesh", "VolumeMesh", "Case"]
        Specifies the highest level of processing to be performed ("SurfaceMesh", "VolumeMesh", or "Case").

    Returns
    -------
    Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
        A tuple containing dictionaries for SurfaceMesh, VolumeMesh, and Case results, if applicable.

    Raises
    ------
    ValueError
        If the private attribute asset cache or project length unit cannot be acquired from the simulation settings.
    """

    params_as_dict = json.loads(simulation_json)
    # Pre-check that project_length_unit exists before validation
    _get_mesh_unit(params_as_dict)

    # Note: There should not be any validation error for params_as_dict. Here is just a deserialization of the JSON
    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.SERVICE,  # This is called only by web service currently.
        root_item_type=root_item_type,
        # Skip context-aware validation for faster performance.
        # Context-aware validation was already performed locally or a priori by backend.
        validation_level=None,
    )

    if errors is not None:
        raise ValueError(str(errors))

    # Extract the validated mesh_unit (a proper unyt quantity) from the params object,
    # not from the raw dict which may be a bare number.
    mesh_unit = params.private_attribute_asset_cache.project_length_unit

    surface_mesh_res = _process_surface_mesh(params, root_item_type, mesh_unit)
    volume_mesh_res = _process_volume_mesh(params, root_item_type, mesh_unit, up_to)
    case_res = _process_case(params, mesh_unit, up_to)

    return surface_mesh_res, volume_mesh_res, case_res


def _convert_unit_in_dict(
    *,
    data: dict,
    target_unit_system: Literal["SI", "Imperial", "CGS"],
    is_delta_temperature: bool,
):

    def get_new_unit_as_string(
        old_unit: u.Unit,
        unit_system: Literal["SI", "Imperial", "CGS"],
        is_delta_temperature: bool,
    ) -> str:
        dimension_str = (
            str(old_unit.dimensions) if not is_delta_temperature else "(temperature_difference)"
        )
        assert (
            dimension_str in supported_units_by_front_end
        ), f"Unknown dimension found: {dimension_str}"

        if isinstance(supported_units_by_front_end[dimension_str], list):
            # This is a unit system agnostic dimension
            for unit in supported_units_by_front_end[dimension_str]:
                if u.Unit(unit) == old_unit:
                    return unit
            return supported_units_by_front_end[dimension_str][0]
        return supported_units_by_front_end[dimension_str][unit_system]

    def get_converted_value(original_value, old_unit, new_unit):
        def convert_single_value(val):
            """Convert a single scalar value with units."""
            converted = (val * old_unit).to(new_unit).value
            # Handle numpy scalars and arrays
            try:
                return float(converted)
            except (TypeError, ValueError):
                # If it's a numpy array or other non-scalar, try to convert to list
                try:
                    return converted.tolist() if hasattr(converted, "tolist") else list(converted)
                except (TypeError, AttributeError):
                    return converted

        def convert_nested_collection(val):
            """Recursively convert nested collections."""
            if isinstance(val, Collection) and not isinstance(val, str):
                if hasattr(val, "__iter__"):
                    return [convert_nested_collection(item) for item in val]
                return convert_single_value(val)
            return convert_single_value(val)

        return convert_nested_collection(original_value)

    old_unit = u.Unit(data["units"])
    new_unit_str = get_new_unit_as_string(
        old_unit, target_unit_system, is_delta_temperature=is_delta_temperature
    )
    new_unit = u.Unit(new_unit_str)
    new_value = get_converted_value(data["value"], old_unit, new_unit)

    data["value"] = new_value
    data["units"] = new_unit_str
    return data


def change_unit_system_recursive(
    *, data, target_unit_system: Literal["SI", "Imperial", "CGS"], current_key: str = None
) -> None:
    """
    Recursively traverse a nested structure of dicts/lists.
    If a dict has exactly the structure {'value': XX, 'units': XX},
    Try to convert to the new unit system
    """
    white_list_keys = {
        # current key -- sub key
        ("private_attribute_asset_cache", "project_length_unit"),
        ("private_attribute_input_cache", "length_unit"),
    }

    if isinstance(data, dict):
        # 1. Check if dict matches the desired pattern
        if set(data.keys()) == {"value", "units"} and data["units"] not in (
            "SI_unit_system",
            "Imperial_unit_system",
            "CGS_unit_system",
        ):
            data = _convert_unit_in_dict(
                data=data,
                target_unit_system=target_unit_system,
                is_delta_temperature=current_key == "temperature_offset",
            )

        # 2. Otherwise, recurse into each item in the dictionary
        for key, val in data.items():
            if (current_key, key) in white_list_keys:
                continue
            change_unit_system_recursive(
                data=val,
                target_unit_system=target_unit_system,
                current_key=key,
            )

    elif isinstance(data, list):
        # Recurse into each item in the list
        for _, item in enumerate(data):
            change_unit_system_recursive(data=item, target_unit_system=target_unit_system)


def change_unit_system(*, data: dict, target_unit_system: Literal["SI", "Imperial", "CGS"]):
    """
    Change the unit system of the simulation parameters.
    """
    change_unit_system_recursive(data=data, target_unit_system=target_unit_system)
    data["unit_system"]["name"] = target_unit_system
    return data


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
