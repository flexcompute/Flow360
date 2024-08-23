"""Simulation services module."""

# pylint: disable=duplicate-code
from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.multi_constructor_model_base import (
    parse_model_dict,
)
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield

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
from flow360.component.simulation.primitives import Box  # For parse_model_dict
from flow360.component.simulation.second_stage_validations.main import (
    destination_validation,
)
from flow360.component.simulation.second_stage_validations.validation_result import (
    ValidationResult,
)
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
from flow360.component.simulation.utils import _model_attribute_unlock
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
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=1, moment_center=(0, 0, 0), moment_length=(1, 1, 1)
            ),
            meshing=MeshingParams(
                refinements=[
                    SurfaceRefinement(
                        name="Global surface refinement", max_edge_length=dummy_value
                    ),
                    BoundaryLayer(
                        name="Global Boundary layer refinement", first_layer_thickness=dummy_value
                    ),
                ],
                volume_zones=[AutomatedFarfield(name="Farfield")],
            ),
            operating_condition=AerospaceCondition(velocity_magnitude=dummy_value),
        )

    if length_unit is not None:
        # Store the length unit so downstream services/pipelines can use it
        # pylint: disable=fixme
        # TODO: client does not call this. We need to start using new webAPI for that
        with _model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
            # pylint: disable=assigning-non-slot,no-member
            params.private_attribute_asset_cache.project_length_unit = LengthType.validate(
                length_unit
            )
    if root_item_type == "Geometry":
        return params.model_dump(
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
                "meshing": {
                    "refinements": {
                        "__all__": {"first_layer_thickness": True, "max_edge_length": True}
                    }
                },
            },
        )
    if root_item_type == "VolumeMesh":
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


class ValidationResponse(pd.BaseModel):
    generic_validtiton: ValidationResult = pd.Field(ValidationResult())
    surface_meshing: ValidationResult = pd.Field(ValidationResult())
    volume_meshing: ValidationResult = pd.Field(ValidationResult())
    case: ValidationResult = pd.Field(ValidationResult())


def _generic_validation(
    params_as_dict, unit_system_name, root_item_type: Literal["Geometry", "VolumeMesh"]
):
    """Validaitons that is destination-agnostic"""
    # To be added when unit system is supported in simulation
    unit_system = init_unit_system(unit_system_name)

    validation_errors = None
    validation_warnings = None
    validated_param = None

    params_as_dict = remove_properties_by_name(params_as_dict, "_id")
    params_as_dict = remove_properties_by_name(params_as_dict, "hash")  #  From client

    if root_item_type == "VolumeMesh":
        params_as_dict.pop("meshing", None)

    try:
        params_as_dict = parse_model_dict(params_as_dict, globals())
        with unit_system:
            validated_param = SimulationParams(**params_as_dict)
    except pd.ValidationError as err:
        validation_errors = err.errors()
    # pylint: disable=broad-exception-caught
    except Exception as err:
        if validation_errors is None:
            validation_errors = []
        # Note: Ideally the definition of WorkbenchValidateWarningOrError should be on the client side?
        validation_errors.append(
            {
                "type": err.__class__.__name__.lower().replace("error", "_error"),
                "loc": ["unknown"],
                "msg": str(err),
                "ctx": {},
            }
        )
        # We do not care about handling / propagating the validation errors here,
        # just collecting them in the context and passing them downstream

    # Check if all validation loc paths are valid params dict paths that can be traversed
    if validation_errors is not None:
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
            try:
                for field_name, field in error["ctx"].items():
                    error["ctx"][field_name] = str(field)
            # pylint: disable=broad-exception-caught
            except Exception:  # This seems to be duplicate info anyway.
                error["ctx"] = {}
    else:
        validation_errors = []

    if validation_warnings is None:
        validation_warnings = []

    return validated_param, ValidationResult(errors=validation_errors, warnings=validation_warnings)


def validate_model(
    params_as_dict,
    unit_system_name,
    root_item_type: Literal["Geometry", "VolumeMesh"],
    run_second_stage_validation: bool = True,
):
    """
    Validate a params dict against the pydantic model
    """
    validated_param, generic_validation_results = _generic_validation(
        params_as_dict, unit_system_name, root_item_type
    )
    ##:: 2nd stage validations
    if generic_validation_results.passed() and run_second_stage_validation is True:
        (
            case_validation_results,
            volume_meshing_validation_results,
            surface_meshing_validation_results,
        ) = destination_validation(root_item_type)

    response = ValidationResponse(
        generic_validtiton=generic_validation_results,
        case=case_validation_results,
        surface_meshing=surface_meshing_validation_results,
        volume_meshing=volume_meshing_validation_results,
    )
    return validated_param, response


# pylint: disable=too-many-arguments
def _translate_simulation_json(
    params_as_dict,
    root_item_type: Literal["Geometry", "VolumeMesh"],
    unit_system_name,
    mesh_unit,
    target_name: str = None,
    translation_func=None,
):
    """
    Get JSON for surface meshing from a given simulaiton JSON.

    """
    translated_dict = None
    # pylint: disable=unused-variable
    param, errors, warnings = validate_model(params_as_dict, unit_system_name, root_item_type)
    if errors is not None:
        # pylint: disable=fixme
        # TODO: Check if this looks good in terminal.
        raise ValueError(errors)
    if mesh_unit is None:
        raise ValueError("Mesh unit is required for translation.")

    try:
        translated_dict = translation_func(param, mesh_unit)
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


def simulation_to_surface_meshing_json(
    params_as_dict, root_item_type: Literal["Geometry", "VolumeMesh"], unit_system_name, mesh_unit
):
    """Get JSON for surface meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        root_item_type,
        unit_system_name,
        mesh_unit,
        "surface meshing",
        get_surface_meshing_json,
    )


def simulation_to_volume_meshing_json(
    params_as_dict, root_item_type: Literal["Geometry", "VolumeMesh"], unit_system_name, mesh_unit
):
    """Get JSON for volume meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        root_item_type,
        unit_system_name,
        mesh_unit,
        "volume meshing",
        get_volume_meshing_json,
    )


def simulation_to_case_json(
    params_as_dict, root_item_type: Literal["Geometry", "VolumeMesh"], unit_system_name, mesh_unit
):
    """Get JSON for case from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        root_item_type,
        unit_system_name,
        mesh_unit,
        "case",
        get_solver_json,
    )
