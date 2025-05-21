"""Resolver and symbols data for Flow360 python client"""

from __future__ import annotations

from typing import Any

import numpy as np
import unyt

from flow360.component.simulation import units as u
from flow360.component.simulation.blueprint.core.resolver import CallableResolver


def _unit_list():
    unit_symbols = set()

    for _, value in unyt.unit_symbols.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            unit_symbols.add(str(value))

    return list(unit_symbols)


def _import_units(_: str) -> Any:
    """Import and return allowed flow360 callables"""
    return u


def _import_numpy(_: str) -> Any:
    """Import and return allowed numpy callables"""
    return np


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
    "numpy": {
        "prefix": "np.",
        "callables": [
            "array",
            "sin",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "dot",
            "cross",
            "sqrt",
        ],
        "evaluate": True,
    },
}

# Define allowed modules
ALLOWED_MODULES = {"u", "np", "control", "solution"}

ALLOWED_CALLABLES = {
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
    "u": _import_units,
    "np": _import_numpy,
}

resolver = CallableResolver(
    ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS, EVALUATION_BLACKLIST
)
