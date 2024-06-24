"""Simulation services module."""

# pylint: disable=duplicate-code
import pydantic as pd

from flow360.component.simulation.models.volume_models import Fluid
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
    SI_unit_system,
    UnitSystem,
    flow360_unit_system,
    imperial_unit_system,
    unit_system_manager,
)

from ..utils import remove_properties_by_name

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


def get_default_params(unit_system_name, length_unit) -> SimulationParams:
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
    if length_unit is not None:
        # TODO implement handling of length_unit (geometry unit and mesh unit), # pylint: disable=fixme
        pass

    unit_system = init_unit_system(unit_system_name)

    with unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=1, moment_center=(0, 0, 0), moment_length=(1, 1, 1)
            ),
            models=[Fluid()],
        )

    return params


def validate_model(params_as_dict, unit_system_name):
    """
    Validate a params dict against the pydantic model
    """

    # To be added when unit system is supported in simulation
    unit_system = init_unit_system(unit_system_name)

    validation_errors = None
    validation_warnings = None
    validated_param = None

    try:
        with unit_system:
            validated_param = SimulationParams(**params_as_dict)
    except pd.ValidationError as err:
        validation_errors = err.errors()
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

    return validated_param, validation_errors, validation_warnings


# pylint: disable=too-many-arguments
def _translate_simulation_json(
    params_as_dict,
    unit_system_name,
    mesh_unit,
    target_name: str = None,
    translation_func=None,
):
    """
    Get JSON for surface meshing from a given simulaiton JSON.

    """
    params_as_dict = remove_properties_by_name(params_as_dict, "_id")

    translated_dict = None
    # pylint: disable=unused-variable
    param, errors, warnings = validate_model(params_as_dict, unit_system_name)
    if errors is not None:
        # pylint: disable=fixme
        # TODO: Check if this looks good in terminal.
        raise ValueError(errors)
    if mesh_unit is None:
        raise ValueError("Mesh unit is required for translation.")
    try:
        translated_dict = translation_func(param, mesh_unit)
    except Exception as err:  # tranlsation itself is not supposed to raise any exception
        raise ValueError(f"Failed to translate to {target_name} json: " + str(err)) from err
    if translated_dict == {}:
        raise ValueError(f"No {target_name} parameters found in given SimulationParams.")
    # pylint: disable=fixme
    # TODO: Implement proper hashing. Currently floating point creates headache for reproducible hashing.
    # pylint: disable=protected-access
    hash_value = SimulationParams._calculate_hash(translated_dict)
    return translated_dict, hash_value


def simulation_to_surface_meshing_json(params_as_dict, unit_system_name, mesh_unit):
    """Get JSON for surface meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        unit_system_name,
        mesh_unit,
        "surface meshing",
        get_surface_meshing_json,
    )


def simulation_to_volume_meshing_json(params_as_dict, unit_system_name, mesh_unit):
    """Get JSON for volume meshing from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        unit_system_name,
        mesh_unit,
        "volume meshing",
        get_volume_meshing_json,
    )


def simulation_to_case_json(params_as_dict, unit_system_name, mesh_unit):
    """Get JSON for case from a given simulaiton JSON."""
    return _translate_simulation_json(
        params_as_dict,
        unit_system_name,
        mesh_unit,
        "case",
        get_solver_json,
    )
