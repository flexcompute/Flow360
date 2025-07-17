"""Context handler module"""

from typing import Any

from unyt import Unit, unit_symbols

from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.blueprint.core.resolver import CallableResolver


def _unit_list():
    """Import a list of available unit symbols from the unyt module"""
    unyt_symbol_dict = {}
    for key, value in unit_symbols.__dict__.items():
        if isinstance(value, Unit):
            dimension_str = str(value.dimensions)
            u_expr_str = str(value.expr)
            if (
                dimension_str.count("logarithmic")
                or dimension_str.count("luminous")
                or dimension_str.count("current")
                or (dimension_str == "1" and u_expr_str != "dimensionless")
            ):
                continue
            if u_expr_str.count("delta_degC") or u_expr_str.count("delta_degF"):
                # Note: Disable the delta temperature units.
                continue
            if u_expr_str not in unyt_symbol_dict:
                unyt_symbol_dict[u_expr_str] = {
                    "aliases": [key],
                    "dimensions": str(value.dimensions),
                    "SI_equivalent": str((1 * value).in_mks()),
                }
            else:
                unyt_symbol_dict[str(value.expr)]["aliases"].append(key)
    allowed_unyt_symbols = []
    for value in unyt_symbol_dict.values():
        allowed_unyt_symbols += value["aliases"]
    return allowed_unyt_symbols


def _import_units(_) -> Any:
    """Import and return allowed unit callables"""
    # pylint:disable=import-outside-toplevel
    from flow360.component.simulation import units as u

    return u


def _import_math(_) -> Any:
    """Import and return allowed function callables"""
    # pylint:disable=import-outside-toplevel, cyclic-import
    from flow360.component.simulation.user_code.functions import math

    return math


def _import_control(_) -> Any:
    """Import and return allowed control variable callables"""
    # pylint:disable=import-outside-toplevel, cyclic-import
    from flow360.component.simulation.user_code.variables import control

    return control


def _import_solution(_) -> Any:
    """Import and return allowed solution variable callables"""
    # pylint:disable=import-outside-toplevel, cyclic-import
    from flow360.component.simulation.user_code.variables import solution

    return solution


WHITELISTED_CALLABLES = {
    "flow360_math": {"prefix": "math.", "callables": ["pi"], "evaluate": True},
    "flow360.units": {"prefix": "u.", "callables": _unit_list(), "evaluate": True},
    "flow360.control": {
        "prefix": "control.",
        "callables": [
            "MachRef",
            "Tref",
            "t",
            "physicalStep",
            "pseudoStep",
            "timeStepSize",
            "alphaAngle",
            "betaAngle",
            "pressureFreestream",
            "momentLengthX",
            "momentLengthY",
            "momentLengthZ",
            "momentCenterX",
            "momentCenterY",
            "momentCenterZ",
            "theta",
            "omega",
            "omegaDot",
        ],
        "evaluate": False,
    },
    "flow360.solution": {
        "prefix": "solution.",
        "callables": [
            # pylint: disable=fixme
            # TODO: Auto-populate this list from the solution module
            "coordinate",
            "Cp",
            "Cpt",
            "grad_density",
            "grad_u",
            "grad_v",
            "grad_w",
            "grad_pressure",
            "Mach",
            "mut",
            "mut_ratio",
            "nu_hat",
            "turbulence_kinetic_energy",
            "specific_rate_of_dissipation",
            "amplification_factor",
            "turbulence_intermittency",
            "density",
            "velocity",
            "pressure",
            "qcriterion",
            "entropy",
            "temperature",
            "vorticity",
            "wall_distance",
            "CfVec",
            "Cf",
            "heat_flux",
            "node_area_vector",
            "node_unit_normal",
            "node_forces_per_unit_area",
            "y_plus",
            "wall_shear_stress_magnitude",
            "heat_transfer_coefficient_static_temperature",
            "heat_transfer_coefficient_total_temperature",
        ],
        "evaluate": False,
    },
}

# Define allowed modules
ALLOWED_MODULES = {"u", "math", "control", "solution"}

ALLOWED_CALLABLES = {
    **{
        f"{group['prefix']}{callable}": None
        for group in WHITELISTED_CALLABLES.values()
        for callable in group["callables"]
    },
}

EVALUATION_BLACKLIST = {
    **{
        f"{group['prefix']}{callable}": None
        for group in WHITELISTED_CALLABLES.values()
        for callable in group["callables"]
        if not group["evaluate"]
    },
}

# Note:  Keys of IMPORT_FUNCTIONS needs to be consistent with ALLOWED_MODULES
IMPORT_FUNCTIONS = {
    "u": _import_units,
    "math": _import_math,
    "control": _import_control,
    "solution": _import_solution,
}

default_context = EvaluationContext(
    CallableResolver(ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS, EVALUATION_BLACKLIST)
)

user_variables: set[str] = set()
solver_variable_name_map: dict[str, str] = {}
