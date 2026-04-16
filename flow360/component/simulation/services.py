"""Simulation services module."""

# pylint: disable=duplicate-code, too-many-lines
import copy
import json
import os
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import pydantic as pd
from flow360_schema.framework.physical_dimensions import Angle, Length
from flow360_schema.models.simulation.services import (  # pylint: disable=unused-import
    ValidationCalledBy,
    _determine_validation_level,
    _insert_forward_compatibility_notice,
    _intersect_validation_levels,
    _normalize_union_branch_error_location,
    _populate_error_context,
    _sanitize_stack_trace,
    _traverse_error_location,
    clean_unrelated_setting_from_params_dict,
    clear_context,
    handle_generic_exception,
    initialize_variable_space,
    validate_error_locations,
    validate_model as _schema_validate_model,
)
from pydantic import TypeAdapter

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.entity_info import (
    merge_geometry_entity_info as merge_geometry_entity_info_obj,
)
from flow360.component.simulation.entity_info import parse_entity_info_model
from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.bet.bet_translator_interface import (
    generate_polar_file_name_list,
    translate_xfoil_c81_to_bet_dict,
    translate_xrotor_dfdc_to_bet_dict,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall

# pylint: disable=unused-import # For parse_model_dict
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import Box

# pylint: enable=unused-import
from flow360.component.simulation.simulation_params import (
    ReferenceGeometry,
    SimulationParams,
)

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
from flow360.component.simulation.unit_system import (
    _UNIT_SYSTEMS,
    UnitSystem,
    _dimensioned_type_serializer,
    u,
    unit_system_manager,
)
from flow360.component.simulation.units import validate_length
from flow360.component.simulation.validation.validation_context import ALL
from flow360.exceptions import (
    Flow360RuntimeError,
    Flow360TranslationError,
    Flow360ValueError,
)
from flow360.version import __version__

# Required for correct global scope initialization


def init_unit_system(unit_system_name) -> UnitSystem:
    """Returns UnitSystem object from string representation.

    Parameters
    ----------
    unit_system_name : ["SI", "CGS", "Imperial"]
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

    unit_system = _UNIT_SYSTEMS.get(unit_system_name)
    if unit_system is None:
        raise ValueError(
            f"Unknown unit system: {unit_system_name!r}. " f"Available: {list(_UNIT_SYSTEMS)}"
        )

    if unit_system_manager.current is not None:
        raise RuntimeError(
            f"Services cannot be used inside unit system context. Used: {unit_system_manager.current.system_repr()}."
        )
    return unit_system


def _store_project_length_unit(project_length_unit, params: SimulationParams):
    if project_length_unit is not None:
        # Store the length unit so downstream services/pipelines can use it
        # pylint: disable=fixme
        # TODO: client does not call this. We need to start using new webAPI for that
        # pylint: disable=assigning-non-slot,no-member
        params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
            "project_length_unit", project_length_unit
        )
    return params


def _get_default_reference_geometry(length_unit: Length.Float64):
    return ReferenceGeometry(
        area=1 * length_unit**2,
        moment_center=(0, 0, 0) * length_unit,
        moment_length=(1, 1, 1) * length_unit,
    )


def get_default_params(
    unit_system_name, length_unit, root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"]
) -> dict:
    """
    Returns default parameters in a given unit system. The defaults are not correct SimulationParams object as they may
    contain empty required values. When generating default case settings:
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    Parameters
    ----------
    unit_system_name : str
        The name of the unit system to use for parameter initialization.

    Returns
    -------
    dict
        Default parameters for Flow360 simulation stored in a dictionary.

    """
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.outputs.outputs import SurfaceOutput
    from flow360.component.simulation.primitives import Surface

    unit_system = init_unit_system(unit_system_name)
    dummy_value = 0.1
    project_length_unit = validate_length(length_unit)
    with unit_system:
        reference_geometry = _get_default_reference_geometry(project_length_unit)
        operating_condition = AerospaceCondition(velocity_magnitude=dummy_value)
        surface_output = SurfaceOutput(
            name="Surface output",
            entities=[Surface(name="*")],
            output_fields=["Cp", "yPlus", "Cf", "CfVec"],
        )

    if root_item_type in ("Geometry", "SurfaceMesh"):
        automated_farfield = AutomatedFarfield(name="Farfield")
        with unit_system:
            params = SimulationParams(
                reference_geometry=reference_geometry,
                meshing=MeshingParams(
                    volume_zones=[automated_farfield],
                ),
                operating_condition=operating_condition,
                models=[
                    Wall(
                        name="Wall",
                        surfaces=[Surface(name="*")],
                        roughness_height=0 * project_length_unit,
                    ),
                    Freestream(name="Freestream", surfaces=[automated_farfield.farfield]),
                ],
                outputs=[surface_output],
            )

        params = _store_project_length_unit(project_length_unit, params)

        return params.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
                "meshing": {
                    "defaults": {"edge_split_layers": True}
                },  # Due to beta mesher by default is disabled.
            },
        )

    if root_item_type == "VolumeMesh":
        with unit_system:
            params = SimulationParams(
                reference_geometry=reference_geometry,
                operating_condition=operating_condition,
                models=[
                    Wall(
                        name="Wall",
                        surfaces=[Surface(name="placeholder1")],
                        roughness_height=0 * project_length_unit,
                    ),  # to make it consistent with geo
                    Freestream(
                        name="Freestream", surfaces=[Surface(name="placeholder2")]
                    ),  # to make it consistent with geo
                ],
                outputs=[surface_output],
            )
        # cleaning up stored entities in default settings to let user decide:
        params.models[0].entities.stored_entities = []  # pylint: disable=unsubscriptable-object
        params.models[1].entities.stored_entities = []  # pylint: disable=unsubscriptable-object

        params = _store_project_length_unit(project_length_unit, params)

        return params.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
                "meshing": True,
            },
        )
    raise ValueError(
        f"Unknown root item type: {root_item_type}. Expected one of Geometry or SurfaceMesh or VolumeMesh"
    )


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


def update_simulation_json(*, params_as_dict: dict, target_python_api_version: str):
    """
    Run the SimulationParams' updater to update to specified version.
    """
    errors = []
    updated_params_as_dict: dict = None
    try:
        # pylint:disable = protected-access
        updated_params_as_dict, input_has_higher_version = SimulationParams._update_param_dict(
            params_as_dict, target_python_api_version
        )
        if input_has_higher_version:
            raise ValueError(
                f"[Internal] API misuse. Input version "
                f"({SimulationParams._get_version_from_dict(model_dict=params_as_dict)}) is higher than "
                f"requested target version ({target_python_api_version})."
            )
    except (Flow360RuntimeError, ValueError, KeyError) as e:
        # Expected exceptions
        errors.append(str(e))
    return updated_params_as_dict, errors


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


def _parse_root_item_type_from_simulation_json(*, param_as_dict: dict):
    """[External] Deduct the root item entity type from simulation.json"""
    try:
        entity_info_type = param_as_dict["private_attribute_asset_cache"]["project_entity_info"][
            "type_name"
        ]
        if entity_info_type == "GeometryEntityInfo":
            return "Geometry"
        if entity_info_type == "SurfaceMeshEntityInfo":
            return "SurfaceMesh"
        if entity_info_type == "VolumeMeshEntityInfo":
            return "VolumeMesh"
        raise ValueError(f"[INTERNAL] Invalid type of the entity info found: {entity_info_type}")
    except KeyError:
        # pylint:disable = raise-missing-from
        raise ValueError("[INTERNAL] Failed to get the root item from the simulation.json!!!")


def merge_geometry_entity_info(
    draft_param_as_dict: dict, geometry_dependencies_param_as_dict: list[dict]
):
    """
    Merge the geometry entity info from geometry dependencies into the draft simulation param dict.

    Parameters
    ----------
    draft_param_as_dict : dict
        The draft simulation parameters dictionary.
    geometry_dependencies_param_as_dict : list of dict
        The list of geometry dependencies simulation parameters dictionaries.

    Returns
    -------
    dict
        The updated draft simulation parameters dictionary with merged geometry entity info.
    """
    draft_param_entity_info_dict = draft_param_as_dict.get("private_attribute_asset_cache", {}).get(
        "project_entity_info", {}
    )
    if draft_param_entity_info_dict.get("type_name") != "GeometryEntityInfo":
        return draft_param_as_dict

    current_entity_info = GeometryEntityInfo.deserialize(draft_param_entity_info_dict)

    entity_info_components = []
    for geometry_param_as_dict in geometry_dependencies_param_as_dict:
        dependency_entity_info_dict = geometry_param_as_dict.get(
            "private_attribute_asset_cache", {}
        ).get("project_entity_info", {})
        if dependency_entity_info_dict.get("type_name") != "GeometryEntityInfo":
            continue
        entity_info_components.append(GeometryEntityInfo.deserialize(dependency_entity_info_dict))

    merged_entity_info = merge_geometry_entity_info_obj(
        current_entity_info=current_entity_info,
        entity_info_components=entity_info_components,
    )
    merged_entity_info_dict = merged_entity_info.model_dump(mode="json", exclude_none=True)

    return merged_entity_info_dict


# Draft entity type names that should be preserved during entity replacement.
# Draft entities (Box, Cylinder, etc.) are user-defined and not tied to uploaded files,
# so they should be kept from the source simulation settings.
# Ghost entities are associated with the geometry/mesh and should be replaced with target's.
def _get_draft_entity_type_names() -> set:
    """Extract entity type names from DraftEntityTypes in entity_info.py."""
    # pylint: disable=import-outside-toplevel
    import types
    from typing import get_args, get_origin

    from flow360.component.simulation.entity_info import EntityInfoModel

    type_names = set()

    # Get draft_entities field type
    draft_field = EntityInfoModel.model_fields[  # pylint:disable=unsubscriptable-object
        "draft_entities"
    ]
    draft_annotation = draft_field.annotation
    # Unwrap List[Annotated[Union[...], ...]] -> Union[...]
    inner_type = get_args(draft_annotation)[0]  # Get inner type from List
    union_args = get_args(inner_type)  # Get Annotated args
    if union_args:
        actual_union = union_args[0]  # First arg is the Union
        # Support both typing.Union and types.UnionType (X | Y syntax in Python 3.10+)
        if get_origin(actual_union) is Union or isinstance(actual_union, types.UnionType):
            for cls in get_args(actual_union):
                type_names.add(cls.__name__)

    return type_names


DRAFT_ENTITY_TYPE_NAMES = _get_draft_entity_type_names()


def _replace_entities_by_type_and_name(
    template_dict: dict,
    target_registry: EntityRegistry,
) -> Tuple[dict, List[Dict[str, Any]]]:
    """
    Traverse template_dict and replace stored_entities with matching entities from target_registry.

    Matching strategy:
    - Use private_attribute_entity_type_name (e.g., "Surface", "Edge") to determine type
    - Use name field for name matching
    - Draft entity types (Box, Cylinder, etc.) are preserved without matching since they are
      user-defined and not tied to uploaded files
    - Ghost and persistent entity types are matched and replaced

    Parameters
    ----------
    template_dict : dict
        The simulation settings dictionary to process
    target_registry : EntityRegistry
        Registry containing target entities for replacement

    Returns
    -------
    Tuple[dict, List[Dict[str, Any]]]
        (Updated dictionary, List of warnings for unmatched entities)
    """
    warnings = []

    # Pre-build lookup dictionary for performance: {(type_name, name): entity_dict}
    entity_lookup: Dict[Tuple[str, str], dict] = {}
    for entity_list in target_registry.internal_registry.values():
        for entity in entity_list:
            key = (entity.private_attribute_entity_type_name, entity.name)
            entity_lookup[key] = entity.model_dump(mode="json", exclude_none=True)

    def process_stored_entities(stored_entities: list) -> list:
        """Process stored_entities list, replacing or removing entities."""
        new_stored = []
        for entity_dict in stored_entities:
            entity_type_name = entity_dict.get("private_attribute_entity_type_name")
            entity_name = entity_dict.get("name")

            # Preserve Draft types directly (user-defined, not tied to uploaded files)
            if entity_type_name in DRAFT_ENTITY_TYPE_NAMES:
                new_stored.append(entity_dict)
                continue

            # Persistent types need matching replacement
            key = (entity_type_name, entity_name)
            if key in entity_lookup:
                new_stored.append(entity_lookup[key])
            else:
                # Skip unmatched entities, record warning
                warnings.append(
                    {
                        "type": "entity_not_found",
                        "loc": ["stored_entities"],
                        "msg": f"Entity '{entity_name}' (type: {entity_type_name}) not found in target entity info",
                        "ctx": {},
                    }
                )
        return new_stored

    def traverse_and_replace(obj):
        """Recursively traverse dict/list to find and process stored_entities."""
        if isinstance(obj, dict):
            if "stored_entities" in obj and isinstance(obj["stored_entities"], list):
                obj["stored_entities"] = process_stored_entities(obj["stored_entities"])
            for value in obj.values():
                traverse_and_replace(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse_and_replace(item)

    traverse_and_replace(template_dict)
    return template_dict, warnings


def apply_simulation_setting_to_entity_info(  # pylint:disable=too-many-locals
    simulation_setting_dict: dict,
    entity_info_dict: dict,
):
    """
    Apply simulation settings from one project to another project with different entity info.

    This function merges simulation settings (case/meshing configuration) from a source
    simulation.json with the entity info from a target simulation.json. It handles entity
    matching by type and name, preserving user-defined draft entities while replacing
    persistent and ghost entities with those from the target.

    Parameters
    ----------
    simulation_setting_dict : dict
        A simulation.json dictionary that provides case/meshing settings.
        This is the "source" that contains the settings to be applied.
    entity_info_dict : dict
        A simulation.json dictionary that provides the entity info (surfaces, edges, etc.).
        This is the "target" whose entities will be used in the result.

    Returns
    -------
    Tuple[dict, Optional[List], List[Dict[str, Any]]]
        A tuple containing:
        - result_dict: The merged simulation.json dictionary
        - errors: List of validation errors, or None if validation passed
        - warnings: List of warnings (unmatched entities + validation warnings)

    Notes
    -----
    Entity handling:
    - Persistent entities (Surface, Edge, GenericVolume, etc.): Matched by (type, name)
      and replaced with target's entities. Unmatched entities are removed with warnings.
    - Draft entities (Box, Cylinder, Point, etc.): Preserved from source without matching,
      as they are user-defined and not tied to uploaded files.
    - Ghost entities: Replaced with target's, as they are associated with the geometry/mesh.

    For GeometryEntityInfo, grouping tags (face_group_tag, body_group_tag, edge_group_tag)
    are inherited from the source to ensure consistent entity selection.
    """
    # pylint:disable=protected-access
    # Step 1: Preprocess both input dicts
    simulation_setting_dict = SimulationParams._sanitize_params_dict(simulation_setting_dict)
    simulation_setting_dict, _ = SimulationParams._update_param_dict(simulation_setting_dict)
    entity_info_dict = SimulationParams._sanitize_params_dict(entity_info_dict)
    entity_info_dict, _ = SimulationParams._update_param_dict(entity_info_dict)

    # Step 2: Extract entity_info from both dicts
    target_entity_info_data = entity_info_dict.get("private_attribute_asset_cache", {}).get(
        "project_entity_info", {}
    )
    source_entity_info = simulation_setting_dict.get("private_attribute_asset_cache", {}).get(
        "project_entity_info", {}
    )

    # Step 3: Merge entity_info (use target's persistent entities, preserve source's draft entities)
    merged_entity_info = copy.deepcopy(target_entity_info_data)
    # Preserve draft entities from source (user-defined, not tied to uploaded files)
    # Ghost entities stay from target as they are associated with the geometry/mesh
    merged_entity_info["draft_entities"] = source_entity_info.get("draft_entities", [])
    # Preserve grouping tags from source (only for GeometryEntityInfo)
    # This ensures the registry is built with the correct grouping selection
    # Only copy grouping tags if target is also GeometryEntityInfo to avoid invalid keys
    if target_entity_info_data.get("type_name") == "GeometryEntityInfo":
        # Map each tag to its corresponding attribute_names field
        tag_to_attr_names = {
            "face_group_tag": "face_attribute_names",
            "body_group_tag": "body_attribute_names",
            "edge_group_tag": "edge_attribute_names",
        }
        for tag_key, attr_names_key in tag_to_attr_names.items():
            source_tag = source_entity_info.get(tag_key)
            if source_tag is not None:
                # Only use source's tag if it exists in target's attribute_names
                # Otherwise keep target's tag to avoid empty registry
                target_attr_names = target_entity_info_data.get(attr_names_key, [])
                if source_tag in target_attr_names:
                    merged_entity_info[tag_key] = source_tag
                # else: keep target's original tag (already in merged_entity_info from deepcopy)

    # Step 4: Build registry from merged entity_info (with source's grouping tags)
    merged_entity_info_obj = parse_entity_info_model(merged_entity_info)
    target_registry = EntityRegistry.from_entity_info(merged_entity_info_obj)

    # Update simulation_setting_dict with merged entity_info
    simulation_setting_dict["private_attribute_asset_cache"][
        "project_entity_info"
    ] = merged_entity_info

    # Step 5: Replace entities in stored_entities
    simulation_setting_dict, entity_warnings = _replace_entities_by_type_and_name(
        simulation_setting_dict, target_registry
    )

    # Step 6: Validate and return results
    root_item_type = _parse_root_item_type_from_simulation_json(
        param_as_dict=simulation_setting_dict
    )
    _, errors, validation_warnings = validate_model(
        params_as_dict=copy.deepcopy(simulation_setting_dict),
        validated_by=ValidationCalledBy.SERVICE,
        root_item_type=root_item_type,
        validation_level=ALL,
    )

    all_warnings = entity_warnings + validation_warnings
    return simulation_setting_dict, errors, all_warnings
