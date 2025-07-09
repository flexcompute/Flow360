"""Utility functions for the user code module"""

import re
from numbers import Number

import numpy as np
import pydantic as pd
from pydantic_core import InitErrorDetails
from unyt import unyt_array, unyt_quantity


def is_number_string(s: str) -> bool:
    """Check if the string represents a single scalar number"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def split_keep_delimiters(value: str, delimiters: list) -> list:
    """split string but keep the delimiters"""
    escaped_delimiters = [re.escape(d) for d in delimiters]
    pattern = f"({'|'.join(escaped_delimiters)})"
    result = re.split(pattern, value)
    return [part for part in result if part != ""]


def handle_syntax_error(se: SyntaxError, source: str):
    """Handle expression syntax error."""
    caret = " " * (se.offset - 1) + "^" if se.text and se.offset else None
    msg = f"{se.msg} at line {se.lineno}, column {se.offset}"
    if caret:
        msg += f"\n{se.text.rstrip()}\n{caret}"

    raise pd.ValidationError.from_exception_data(
        "expression_syntax",
        [
            InitErrorDetails(
                type="value_error",
                msg=se.msg,
                input=source,
                ctx={
                    "line": se.lineno,
                    "column": se.offset,
                    "error": msg,
                },
            )
        ],
    )


def is_runtime_expression(value):
    """Check if the input value is a runtime expression."""
    if isinstance(value, unyt_quantity) and np.isnan(value.value):
        return True
    if isinstance(value, unyt_array) and np.isnan(value.value).any():
        return True
    if isinstance(value, Number) and np.isnan(value):
        return True
    if isinstance(value, list) and any(np.isnan(item) for item in value):
        return any(np.isnan(item) for item in value)
    return False


SOLVER_INTERNAL_VARIABLES = {
    "bet_omega",
    "bet_torque",
    "bet_thrust",
    "coordinate",
    "primitiveVars",
    "gradPrimitive",
    "wallShearStress",
    "wallViscousStress",
    "nodeNormals",
    "t",
    "mut",
    "mu",
    "primitiveNonInertial",
    "gradPrimitiveNonInertial",
    "wallDistance",
    "solutionNavierStokes",
    "residualNavierStokes",
    "solutionTransition",
    "residualTransition",
    "solutionHeatSolver",
    "residualHeatSolver",
    "theta",
    "omega",
    "omegaDot",
    "previousTheta",
    "yPlus",
    "heatFlux",
    "CL",
    "CD",
    "momentX",
    "momentY",
    "momentZ",
    "forceX",
    "forceY",
    "forceZ",
    "wallFunctionMetric",
    "massFlowRate",
    "staticPressureRatio",
    "totalPressureRatio",
    "area",
    "hasSupersonicFlow",
}
