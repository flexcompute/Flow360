"""Service-level entrypoints for the simulation translator.

These wrappers accept a validated SimulationParams and delegate to the
underlying solver/meshing translation functions, handling error translation
to ValueError and hash computation. Separated from services.py so
translator-specific code can live alongside the translator package (and
eventually move to compute with it).
"""

from flow360_schema.exceptions import Flow360TranslationError
from flow360_schema.models.simulation.simulation_params import SimulationParams

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


# pylint: disable=too-many-arguments
def _translate_simulation_json(
    input_params: SimulationParams,
    mesh_unit,
    target_name: str = None,
    translation_func=None,
    **kwargs,
):
    """
    Run a translation function against SimulationParams and wrap errors.
    """
    translated_dict = None
    if mesh_unit is None:
        raise ValueError("Mesh unit is required for translation.")
    if isinstance(input_params, SimulationParams) is False:
        raise ValueError(
            "input_params must be of type SimulationParams. Instead got: "
            + str(type(input_params))
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
