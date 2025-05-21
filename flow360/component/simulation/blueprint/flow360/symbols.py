from __future__ import annotations

from typing import Any

from flow360.component.simulation.blueprint.core.resolver import CallableResolver


def _unit_list():
    import unyt

    unit_symbols = set()

    for key, value in unyt.unit_symbols.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            unit_symbols.add(str(value))

    return list(unit_symbols)


def _import_flow360(name: str) -> Any:
    """Import and return a flow360 callable"""

    if name == "u":
        from flow360.component.simulation import units as u

        return u


def _import_numpy(name: str) -> Any:
    import numpy as np

    if name == "np":
        return np


# TODO: Rename the variables to conform to the snake case convention
WHITELISTED_CALLABLES = {
    "flow360.units": {"prefix": "u.", "callables": _unit_list(), "evaluate": True},
    "flow360.control": {
        "prefix": "control.",
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
            "theta",
            "omega",
            "omegaDot",
            "wallFunctionMetric",
            "wallShearStress",
            "yPlus",
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
            "theta",
            "omega",
            "omegaDot",
            "wallFunctionMetric",
            "wallShearStress",
            "yPlus",
        ],
        "evaluate": False,
    },
}

# Define allowed modules
ALLOWED_MODULES = {"flow360", "fl", "np"}

ALLOWED_CALLABLES = {
    "fl": None,
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
    },
}

EVALUATION_BLACKLIST = {
    **{
        f"{group['prefix']}{name}": None
        for group in WHITELISTED_CALLABLES.values()
        for name in group["callables"]
        if not group["evaluate"]
    },
}

IMPORT_FUNCTIONS = {
    ("fl", "u"): _import_flow360,
    "np": _import_numpy,
}

resolver = CallableResolver(
    ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS, EVALUATION_BLACKLIST
)
