"""Simulation services module."""

# pylint: disable=duplicate-code
import json
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import Freestream, Wall

# pylint: disable=unused-import
from flow360.component.simulation.operating_condition.operating_condition import (
    GenericReferenceCondition,  # For parse_model_dict
)
from flow360.component.simulation.operating_condition.operating_condition import (
    ThermalState,  # For parse_model_dict
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Box  # For parse_model_dict
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import (
    ReferenceGeometry,
    SimulationParams,
)
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import (
    CGS_unit_system,
    LengthType,
    SI_unit_system,
    UnitSystem,
    flow360_unit_system,
    imperial_unit_system,
    unit_system_manager,
)
from flow360.component.simulation.utils import (
    get_unit_system_name_from_simulation_params_dict,
    model_attribute_unlock,
)
from flow360.component.simulation.validation.validation_context import (
    ALL,
    SURFACE_MESH,
    VOLUME_MESH,
    ValidationLevelContext,
)
from flow360.component.utils import remove_properties_by_name
from flow360.exceptions import Flow360TranslationError

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


def _store_project_length_unit(length_unit, params: SimulationParams):
    if length_unit is not None:
        # Store the length unit so downstream services/pipelines can use it
        # pylint: disable=fixme
        # TODO: client does not call this. We need to start using new webAPI for that
        with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
            # pylint: disable=assigning-non-slot,no-member
            params.private_attribute_asset_cache.project_length_unit = LengthType.validate(
                length_unit
            )
    return params


def get_default_params(
    unit_system_name, length_unit, root_item_type: Literal["Geometry", "VolumeMesh"]
) -> SimulationParams:
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
    SimulationParams
        Default parameters for Flow360 simulation.

    """

    unit_system = init_unit_system(unit_system_name)
    dummy_value = 0.1
    with unit_system:
        reference_geometry = ReferenceGeometry(
            area=1, moment_center=(0, 0, 0), moment_length=(1, 1, 1)
        )
        operating_condition = AerospaceCondition(velocity_magnitude=dummy_value)
        surface_output = SurfaceOutput(
            name="Surface output",
            entities=[Surface(name="*")],
            output_fields=["Cp", "yPlus", "Cf", "CfVec"],
        )

    if root_item_type == "Geometry":
        automated_farfield = AutomatedFarfield(name="Farfield")
        with unit_system:
            params = SimulationParams(
                reference_geometry=reference_geometry,
                meshing=MeshingParams(
                    volume_zones=[automated_farfield],
                ),
                operating_condition=operating_condition,
                models=[
                    Wall(name="Wall", surfaces=[Surface(name="*")]),
                    Freestream(name="Freestream", surfaces=[automated_farfield.farfield]),
                ],
                outputs=[surface_output],
            )

        params = _store_project_length_unit(length_unit, params)

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
                        name="Wall", surfaces=[Surface(name="placeholder1")]
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

        params = _store_project_length_unit(length_unit, params)

        return params.model_dump(
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
                "meshing": True,
            },
        )
    raise ValueError(
        f"Unknown root item type: {root_item_type}. Expected one of Geometry or VolumeMesh"
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


def validate_model(
    *,
    params_as_dict,
    root_item_type: Union[Literal["Geometry", "VolumeMesh"], None],
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
    root_item_type : Union[Literal["Geometry", "VolumeMesh"], None],
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
    unit_system_name = get_unit_system_name_from_simulation_params_dict(params_as_dict)
    unit_system = init_unit_system(
        unit_system_name
    )  # Initialize unit system (to be implemented when supported)

    validation_errors = None
    validation_warnings = None
    validated_param = None

    params_as_dict = clean_params_dict(params_as_dict, root_item_type)

    # The final validation levels will be the intersection of the requested levels and the levels available
    # We always assume we want to run case so that we can expose as many errors as possible
    available_levels = _determine_validation_level(up_to="Case", root_item_type=root_item_type)
    validation_levels_to_use = _intersect_validation_levels(validation_level, available_levels)
    try:
        params_as_dict = parse_model_dict(params_as_dict, globals())
        with unit_system:
            with ValidationLevelContext(validation_levels_to_use):
                validated_param = SimulationParams(**params_as_dict)
    except pd.ValidationError as err:
        validation_errors = err.errors()
    except Exception as err:  # pylint: disable=broad-exception-caught
        validation_errors = handle_generic_exception(err, validation_errors)

    if validation_errors is not None:
        validation_errors = validate_error_locations(validation_errors, params_as_dict)

    return validated_param, validation_errors, validation_warnings


def clean_params_dict(params: dict, root_item_type: str) -> dict:
    """
    Cleans the parameters dictionary by removing unwanted properties.

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
    params = remove_properties_by_name(params, "_id")
    params = remove_properties_by_name(params, "hash")  # From client

    if root_item_type == "VolumeMesh":
        params.pop("meshing", None)

    return params


def handle_generic_exception(err: Exception, validation_errors: Optional[list]) -> list:
    """
    Handles generic exceptions during validation, adding to validation errors.

    Parameters
    ----------
    err : Exception
        The exception caught during validation.
    validation_errors : list or None
        Current list of validation errors, may be None.

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
            "loc": ["unknown"],
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
    Get JSON for surface meshing from a given simulaiton JSON.

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
    except Exception as err:  # tranlsation itself is not supposed to raise any other exception
        raise ValueError(
            f"Unexpected error translating to {target_name} json: " + str(err)
        ) from err

    if translated_dict == {}:
        raise ValueError(f"No {target_name} parameters found in given SimulationParams.")

    # pylint: disable=protected-access
    hash_value = SimulationParams._calculate_hash(translated_dict)
    return translated_dict, hash_value


def simulation_to_surface_meshing_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for surface meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "surface meshing",
        get_surface_meshing_json,
    )


def simulation_to_volume_meshing_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for volume meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        input_params,
        mesh_unit,
        "volume meshing",
        get_volume_meshing_json,
    )


def simulation_to_case_json(input_params: SimulationParams, mesh_unit):
    """Get JSON for case from a given simulaiton JSON."""
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
    root_item_type: Union[Literal["Geometry", "VolumeMesh"], None],
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
    root_item_type: Literal["Geometry", "VolumeMesh"],
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
    root_item_type : Literal["Geometry", "VolumeMesh"]
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
        root_item_type=root_item_type,
        validation_level=validation_level,
    )

    if errors is not None:
        raise ValueError(
            "[Internal] Validation error occurred for supposedly validated param! Errors are: "
            + str(errors)
        )

    surface_mesh_res = _process_surface_mesh(params, root_item_type, mesh_unit)
    volume_mesh_res = _process_volume_mesh(params, root_item_type, mesh_unit, up_to)
    case_res = _process_case(params, mesh_unit, up_to)

    return surface_mesh_res, volume_mesh_res, case_res


def change_unit_system(
    *, simulation_params: SimulationParams, target_unit_system: Literal["SI", "Imperial", "CGS"]
):
    """
    Changes the unit system of the simulation parameters and convert the values accordingly.
    """
    converted_params = simulation_params.convert_to_unit_system(unit_system=target_unit_system)
    return converted_params.model_dump_json(exclude_none=True)
