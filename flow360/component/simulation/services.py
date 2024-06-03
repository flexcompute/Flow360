import pydantic as pd

from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import (
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    flow360_unit_system,
    imperial_unit_system,
    unit_system_manager,
)

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
        raise ValueError(f"Incorrect unit system provided {unit_system=}, expected type UnitSystem")

    if unit_system_manager.current is not None:
        raise RuntimeError(
            f"Services cannot be used inside unit system context. Used: {unit_system_manager.current.system_repr()}."
        )
    return unit_system


def validate_model(params_as_dict, unit_system_name):
    """
    Validate a params dict against the pydantic model
    """

    # To be added when unit system is supported in simulation
    # unit_system = init_unit_system(unit_system_name)
    # params_as_dict["unitSystem"] = unit_system.dict()

    validation_errors = None

    try:
        params = SimulationParams(**params_as_dict)
    except pd.ValidationError as err:
        validation_errors = err.errors()
        # We do not care about handling / propagating the validation errors here,
        # just collecting them in the context and passing them downstream
        pass

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

    return validation_errors
