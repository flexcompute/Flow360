from typing import Any

from flow360.component.simulation.blueprint.core import EvaluationContext
from flow360.component.simulation.blueprint.core.resolver import CallableResolver


def _unit_list():
    """Import a list of available unit symbols from the unyt module"""
    from unyt import Unit, unit_symbols, unyt_array

    symbols = set()

    for _, value in unit_symbols.__dict__.items():
        if isinstance(value, (unyt_array, Unit)):
            symbols.add(str(value))

    return list(symbols)


def _import_units(_) -> Any:
    """Import and return allowed unit callables"""
    from flow360.component.simulation import units as u

    return u


def _import_math(_) -> Any:
    """Import and return allowed function callables"""
    from flow360.component.simulation.user_code.functions import math

    return math


def _import_control(_) -> Any:
    """Import and return allowed control variable callables"""
    from flow360.component.simulation.user_code.variables import control

    return control


def _import_solution(_) -> Any:
    """Import and return allowed solution variable callables"""
    from flow360.component.simulation.user_code.variables import solution

    return solution


WHITELISTED_CALLABLES = {
    "flow360_math": {"prefix": "fn.", "callables": ["cross"], "evaluate": True},
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
            "mut",
            "mu",
            "solutionNavierStokes",
            "residualNavierStokes",
            "solutionTurbulence",
            "residualTurbulence",
            "kOmega",
            "nuHat",
            "solutionTransition",
            "residualTransition",
            "solutionHeatSolver",
            "residualHeatSolver",
            "coordinate",
            "velocity",
            "bet_thrust",
            "bet_torque",
            "bet_omega",
            "CD",
            "CL",
            "forceX",
            "forceY",
            "forceZ",
            "momentX",
            "momentY",
            "momentZ",
            "nodeNormals",
            "wallFunctionMetric",
            "wallShearStress",
            "yPlus",
        ],
        "evaluate": False,
    },
}

# Define allowed modules
ALLOWED_MODULES = {"u", "fl", "control", "solution", "math"}

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
