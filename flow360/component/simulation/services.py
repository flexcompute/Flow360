"""Simulation services module."""

# pylint: disable=duplicate-code, too-many-lines
import json
import os
from enum import Enum
from typing import Any, Collection, Dict, Iterable, Literal, Optional, Tuple, Union

import pydantic as pd
from pydantic_core import ErrorDetails

# Required for correct global scope initialization
from flow360.component.simulation.blueprint.core.dependency_graph import DependencyGraph
from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.bet.bet_translator_interface import (
    generate_polar_file_name_list,
    translate_xfoil_c81_to_bet_dict,
    translate_xrotor_dfdc_to_bet_dict,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall

# Following unused-import for supporting parse_model_dict
from flow360.component.simulation.models.volume_models import (  # pylint: disable=unused-import
    BETDisk,
)

# pylint: disable=unused-import
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Box  # pylint: disable=unused-import
from flow360.component.simulation.primitives import Surface  # For parse_model_dict
from flow360.component.simulation.simulation_params import (
    ReferenceGeometry,
    SimulationParams,
)

# Required for correct global scope initialization
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import (
    AngleType,
    CGS_unit_system,
    LengthType,
    SI_unit_system,
    UnitSystem,
    _dimensioned_type_serializer,
    flow360_unit_system,
    imperial_unit_system,
    u,
    unit_system_manager,
)
from flow360.component.simulation.user_code.core.types import (
    UserVariable,
    get_referenced_expressions_and_user_variables,
)
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.validation.validation_context import (
    ALL,
    ParamsValidationInfo,
    ValidationContext,
    get_value_with_path,
)
from flow360.exceptions import (
    Flow360RuntimeError,
    Flow360TranslationError,
    Flow360ValueError,
)
from flow360.plugins.report.report import get_default_report_summary_template
from flow360.version import __version__

# Required for correct global scope initialization


unit_system_map = {
    "SI": SI_unit_system,
    "CGS": CGS_unit_system,
    "Imperial": imperial_unit_system,
    "Flow360": flow360_unit_system,
}


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


def _store_project_length_unit(project_length_unit, params: SimulationParams):
    if project_length_unit is not None:
        # Store the length unit so downstream services/pipelines can use it
        # pylint: disable=fixme
        # TODO: client does not call this. We need to start using new webAPI for that
        with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
            # pylint: disable=assigning-non-slot,no-member
            params.private_attribute_asset_cache.project_length_unit = project_length_unit
    return params


def _get_default_reference_geometry(length_unit: LengthType):
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

    unit_system = init_unit_system(unit_system_name)
    dummy_value = 0.1
    project_length_unit = LengthType.validate(length_unit)  # pylint: disable=no-member
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
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
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


def _intersect_validation_levels(requested_levels, available_levels):
    if requested_levels is not None and available_levels is not None:
        if requested_levels == ALL:
            validation_levels_to_use = [
                item for item in ["SurfaceMesh", "VolumeMesh", "Case"] if item in available_levels
            ]
        elif isinstance(requested_levels, str):
            if requested_levels in available_levels:
                validation_levels_to_use = [requested_levels]
            else:
                validation_levels_to_use = None
        else:
            assert isinstance(requested_levels, list)
            validation_levels_to_use = [
                item for item in requested_levels if item in available_levels
            ]
        return validation_levels_to_use
    return None


class ValidationCalledBy(Enum):
    """
    Enum as indicator where `validate_model()` is called.
    """

    LOCAL = "Local"
    SERVICE = "Service"
    PIPELINE = "Pipeline"

    def get_forward_compatibility_error_message(self, version_from: str, version_to: str):
        """
        Return error message string indicating that the forward compatibility is not guaranteed.
        """
        error_suffix = " Errors may occur since forward compatibility is limited."
        if self == ValidationCalledBy.LOCAL:
            return {
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "The cloud `SimulationParam` (version: "
                + version_from
                + ") is too new for your local Python client (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        if self == ValidationCalledBy.SERVICE:
            return {
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "Your `SimulationParams` (version: "
                + version_from
                + ") is too new for the solver (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        if self == ValidationCalledBy.PIPELINE:
            # These will only appear in log. Ideally we should not rely on pipelines
            # to emit useful error messages. Or else the local/service validation is not doing their jobs properly.
            return {
                # pylint:disable = protected-access
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "[Internal] Your `SimulationParams` (version: "
                + version_from
                + ") is too new for the solver (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        return None


def _insert_forward_compatibility_notice(
    validation_errors: list,
    params_as_dict: dict,
    validated_by: ValidationCalledBy,
    version_to: str = __version__,
):
    # If error occurs, inform user that the error message could due to failure in forward compatibility.
    # pylint:disable=protected-access
    version_from = SimulationParams._get_version_from_dict(model_dict=params_as_dict)
    forward_compatibility_failure_error = validated_by.get_forward_compatibility_error_message(
        version_from=version_from, version_to=version_to
    )
    validation_errors.insert(0, forward_compatibility_failure_error)
    return validation_errors


def initialize_variable_space(param_as_dict: dict, use_clear_context: bool = False) -> dict:
    """Load all user variables from private attributes when a simulation params object is initialized"""
    if "private_attribute_asset_cache" not in param_as_dict.keys():
        return param_as_dict
    asset_cache: dict = param_as_dict["private_attribute_asset_cache"]
    if "variable_context" not in asset_cache.keys():
        return param_as_dict
    if not isinstance(asset_cache["variable_context"], Iterable):
        return param_as_dict

    if use_clear_context:
        clear_context()

    # ==== Build dependency graph and sort variables ====
    dependency_graph = DependencyGraph()
    # Pad the project variables into proper schema
    variable_list = []
    for var in asset_cache["variable_context"]:
        if "type_name" in var["value"] and var["value"]["type_name"] == "expression":
            # Expression type
            variable_list.append({"name": var["name"], "value": var["value"]["expression"]})
        else:
            # Number type (#units ignored since it does not affect the dependency graph)
            variable_list.append({"name": var["name"], "value": str(var["value"]["value"])})
    dependency_graph.load_from_list(variable_list)
    sorted_variables = dependency_graph.topology_sort()

    pre_sort_name_to_index = {
        var["name"]: idx for idx, var in enumerate(asset_cache["variable_context"])
    }

    for variable_name in sorted_variables:
        variable_dict = next(
            (var for var in asset_cache["variable_context"] if var["name"] == variable_name),
            None,
        )
        if variable_dict is None:
            continue

        value_or_expression = dict(variable_dict["value"].items())

        try:
            UserVariable(
                name=variable_dict["name"],
                value=value_or_expression,
                **(
                    {"description": variable_dict["description"]}
                    if "description" in variable_dict
                    else {}
                ),
            )
        except pd.ValidationError as e:
            # pylint:disable = raise-missing-from
            if "Redeclaring user variable" in str(e):
                raise ValueError(
                    f"Loading user variable '{variable_dict['name']}' from simulation.json which is "
                    "already defined in local context. Please change your local user variable definition."
                )
            error_detail: dict = e.errors()[0]
            raise pd.ValidationError.from_exception_data(
                "Invalid user variable/expression",
                line_errors=[
                    ErrorDetails(
                        type=error_detail["type"],
                        loc=(
                            "private_attribute_asset_cache",
                            "variable_context",
                            pre_sort_name_to_index[variable_name],
                        ),
                        msg=error_detail["msg"],
                        ctx=error_detail["ctx"],
                    ),
                ],
            )

    return param_as_dict


def validate_model(  # pylint: disable=too-many-locals
    *,
    params_as_dict,
    validated_by: ValidationCalledBy,
    root_item_type: Union[Literal["Geometry", "SurfaceMesh", "VolumeMesh"], None],
    validation_level: Union[
        Literal["SurfaceMesh", "VolumeMesh", "Case", "All"], list, None
    ] = ALL,  # Fix implicit string concatenation
) -> Tuple[Optional[SimulationParams], Optional[list], Optional[list]]:
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
    validation_warnings : list or None
        A list of validation warnings if any occurred.
    """
    validation_errors = None
    validation_warnings = None
    validated_param = None

    params_as_dict = clean_unrelated_setting_from_params_dict(params_as_dict, root_item_type)

    # The final validation levels will be the intersection of the requested levels and the levels available
    # We always assume we want to run case so that we can expose as many errors as possible
    available_levels = _determine_validation_level(up_to="Case", root_item_type=root_item_type)
    validation_levels_to_use = _intersect_validation_levels(validation_level, available_levels)
    forward_compatibility_mode = False

    try:
        # pylint: disable=protected-access
        # Note: Need to run updater first to accommodate possible schema change in input caches.
        updated_param_as_dict, forward_compatibility_mode = SimulationParams._update_param_dict(
            params_as_dict
        )

        use_clear_context = validated_by == ValidationCalledBy.SERVICE
        initialize_variable_space(updated_param_as_dict, use_clear_context)

        referenced_expressions = get_referenced_expressions_and_user_variables(
            updated_param_as_dict
        )

        additional_info = ParamsValidationInfo(
            param_as_dict=updated_param_as_dict,
            referenced_expressions=referenced_expressions,
        )

        with ValidationContext(levels=validation_levels_to_use, info=additional_info):
            # Multi-constructor model support
            updated_param_as_dict = parse_model_dict(updated_param_as_dict, globals())
            validated_param = SimulationParams(file_content=updated_param_as_dict)
    except pd.ValidationError as err:
        validation_errors = err.errors()
    except Exception as err:  # pylint: disable=broad-exception-caught
        validation_errors = handle_generic_exception(err, validation_errors)

    if validation_errors is not None:
        validation_errors = validate_error_locations(validation_errors, params_as_dict)

    if forward_compatibility_mode and validation_errors is not None:
        # pylint: disable=fixme
        # TODO: If forward compatibility issue found. Try to tell user how they can get around it.
        # TODO: Recommend solver/python client version they should use instead.
        validation_errors = _insert_forward_compatibility_notice(
            validation_errors, params_as_dict, validated_by
        )

    return validated_param, validation_errors, validation_warnings


def clean_unrelated_setting_from_params_dict(params: dict, root_item_type: str) -> dict:
    """
    Cleans the parameters dictionary by removing properties if they do not affect the remaining workflow.


    Parameters
    ----------
    params : dict
        The original parameters dictionary.
    root_item_type : str
        The root item type determining specific cleaning actions.

    Returns
    -------
    dict
        The cleaned parameters dictionary.
    """

    if root_item_type == "VolumeMesh":
        params.pop("meshing", None)

    return params


def handle_generic_exception(
    err: Exception, validation_errors: Optional[list], loc_prefix: Optional[list[str]] = None
) -> list:
    """
    Handles generic exceptions during validation, adding to validation errors.

    Parameters
    ----------
    err : Exception
        The exception caught during validation.
    validation_errors : list or None
        Current list of validation errors, may be None.
    loc_prefix : list or None
        Prefix of the location of the generic error to help locate the issue

    Returns
    -------
    list
        The updated list of validation errors including the new error.
    """
    if validation_errors is None:
        validation_errors = []

    validation_errors.append(
        {
            "type": err.__class__.__name__.lower().replace("error", "_error"),
            "loc": ["unknown"] if loc_prefix is None else loc_prefix,
            "msg": str(err),
            "ctx": {},
        }
    )
    return validation_errors


def validate_error_locations(errors: list, params: dict) -> list:
    """
    Validates the locations in the errors to ensure they correspond to the params dict.

    Parameters
    ----------
    errors : list
        The list of validation errors to process.
    params : dict
        The parameters dictionary being validated.

    Returns
    -------
    list
        The updated list of errors with validated locations and context.
    """
    for error in errors:
        current = params
        for field in error["loc"][:-1]:
            current, valid = _traverse_error_location(current, field)
            if not valid:
                error["loc"] = tuple(loc for loc in error["loc"] if loc != field)

        _populate_error_context(error)
    return errors


def _traverse_error_location(current, field):
    """
    Traverse through the error location path within the parameters.

    Parameters
    ----------
    current : any
        The current position in the params dict or list.
    field : any
        The current field being validated.

    Returns
    -------
    tuple
        The updated current position and whether the traversal was valid.
    """
    if isinstance(field, int) and isinstance(current, list) and field in range(len(current)):
        return current[field], True
    if isinstance(field, str) and isinstance(current, dict) and current.get(field):
        return current.get(field), True
    return current, False


def _populate_error_context(error: dict):
    """
    Populates the error context with relevant stringified values.

    Parameters
    ----------
    error : dict
        The error dictionary to update with context information.
    """
    ctx = error.get("ctx")
    if isinstance(ctx, dict):
        for field_name, context in ctx.items():
            try:
                error["ctx"][field_name] = (
                    [str(item) for item in context] if isinstance(context, list) else str(context)
                )
            except Exception:  # pylint: disable=broad-exception-caught
                error["ctx"][field_name] = "<couldn't stringify>"
    else:
        error["ctx"] = {}


# pylint: disable=too-many-arguments
def _translate_simulation_json(
    input_params: SimulationParams,
    mesh_unit,
    target_name: str = None,
    translation_func=None,
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
        translated_dict = translation_func(input_params, mesh_unit)
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


def simulation_to_case_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for case from a given simulation JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "case",
        get_solver_json,
    )


def _get_mesh_unit(params_as_dict: dict) -> str:
    if params_as_dict.get("private_attribute_asset_cache") is None:
        raise ValueError("[Internal] failed to acquire length unit from simulation settings.")
    mesh_unit = params_as_dict["private_attribute_asset_cache"].get("project_length_unit")
    if mesh_unit is None:
        raise ValueError("[Internal] failed to acquire length unit from simulation settings.")
    return mesh_unit


def _determine_validation_level(
    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"],
    root_item_type: Union[Literal["Geometry", "SurfaceMesh", "VolumeMesh"], None],
) -> list:
    if root_item_type is None:
        return None
    all_lvls = ["Geometry", "SurfaceMesh", "VolumeMesh", "Case"]
    return all_lvls[all_lvls.index(root_item_type) + 1 : all_lvls.index(up_to) + 1]


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
    mesh_unit = _get_mesh_unit(params_as_dict)
    validation_level = _determine_validation_level(up_to, root_item_type)

    # Note: There should not be any validation error for params_as_dict. Here is just a deserilization of the JSON
    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.SERVICE,  # This is called only by web service currently.
        root_item_type=root_item_type,
        validation_level=validation_level,
    )

    if errors is not None:
        raise ValueError(str(errors))

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
        if isinstance(original_value, Collection):
            new_value = []
            for value in original_value:
                value = (value * old_unit).to(new_unit).value
                new_value.append(float(value))
        else:
            new_value = float((original_value * old_unit).to(new_unit).value)
        return new_value

    old_unit = u.Unit(data["units"])
    new_unit_str = get_new_unit_as_string(
        old_unit, target_unit_system, is_delta_temperature=is_delta_temperature
    )
    new_unit = u.Unit(new_unit_str)
    new_value = get_converted_value(data["value"], old_unit, new_unit)

    data["value"] = new_value
    data["units"] = new_unit_str
    return data


def change_unit_system(
    *, data, target_unit_system: Literal["SI", "Imperial", "CGS"], current_key: str = None
) -> None:
    """
    Recursively traverse a nested structure of dicts/lists.
    If a dict has exactly the structure {'value': XX, 'units': XX},
    Try to convert to the new unit system
    """

    if isinstance(data, dict):
        # 1. Check if dict matches the desired pattern
        if set(data.keys()) == {"value", "units"}:
            data = _convert_unit_in_dict(
                data=data,
                target_unit_system=target_unit_system,
                is_delta_temperature=current_key == "temperature_offset",
            )

        # 2. Otherwise, recurse into each item in the dictionary
        for key, val in data.items():
            change_unit_system(
                data=val,
                target_unit_system=target_unit_system,
                current_key=key,
            )

    elif isinstance(data, list):
        # Recurse into each item in the list
        for _, item in enumerate(data):
            change_unit_system(data=item, target_unit_system=target_unit_system)


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


def clear_context():
    """
    Clear out `UserVariable` in the `context` and its dependency graph.
    """

    from flow360.component.simulation.user_code.core import (  # pylint: disable=import-outside-toplevel
        context,
    )

    # pylint: disable=protected-access
    for name in context.default_context._values.keys():
        if "." not in name:
            context.default_context._dependency_graph.remove_variable(name)
    context.default_context._values = {
        name: value for name, value in context.default_context._values.items() if "." in name
    }


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


def _validate_unit_string(unit_str: str, unit_type: Union[AngleType, LengthType]) -> bool:
    """
    Validate the unit string from request against the specified unit type.
    """
    try:
        unit_dict = json.loads(unit_str)
        return unit_type.validate(unit_dict)
    except json.JSONDecodeError:
        return unit_type.validate(unit_str)


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
        length_unit = _validate_unit_string(length_unit, LengthType)
        angle_unit = _validate_unit_string(angle_unit, AngleType)
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
        length_unit = _validate_unit_string(length_unit, LengthType)
        angle_unit = _validate_unit_string(angle_unit, AngleType)
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


def get_default_report_config() -> dict:
    """
    Get the default report config
    Returns
    -------
    dict
        default report config
    """
    return get_default_report_summary_template().model_dump(
        exclude_none=True,
    )


def _parse_root_item_type_from_simulation_json(*, param_as_dict: dict):
    """Deduct the root item entity type from simulation.json"""
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
