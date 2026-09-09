"""Expression registry: default_context singleton, whitelist configuration, and solver variable names.

This module is intentionally isolated (Category 6 in the expression module taxonomy)
so that the global default_context can be removed in the future without affecting
other components.
"""

from typing import Any

from unyt import Unit, unit_symbols

from flow360_schema.framework.expression.engine.eval_context import EvaluationContext
from flow360_schema.framework.expression.engine.resolver import CallableResolver

# ---------------------------------------------------------------------------
# Unit symbol whitelist
# ---------------------------------------------------------------------------


def _unit_list() -> list[str]:
    """Generate the list of allowed unyt unit symbols.

    Kept as a function (not hardcoded) because unyt exposes ~2700 symbols
    that vary across versions. Only called once at module init.
    """
    unyt_symbol_dict: dict[str, dict[str, list[str]]] = {}
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
                continue
            if u_expr_str not in unyt_symbol_dict:
                unyt_symbol_dict[u_expr_str] = {"aliases": [key]}
            else:
                unyt_symbol_dict[u_expr_str]["aliases"].append(key)
    allowed_unyt_symbols: list[str] = []
    for value in unyt_symbol_dict.values():
        allowed_unyt_symbols += value["aliases"]
    return allowed_unyt_symbols


_ALLOWED_UNYT_SYMBOLS = _unit_list()


# ---------------------------------------------------------------------------
# Whitelist configuration
# ---------------------------------------------------------------------------

WHITELISTED_CALLABLES: dict[str, dict[str, str | list[str] | bool]] = {
    "flow360_math": {"prefix": "math.", "callables": ["pi"], "evaluate": True},
    "flow360.units": {"prefix": "u.", "callables": _ALLOWED_UNYT_SYMBOLS, "evaluate": True},
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
            "coordinate",
            "Cp",
            "Cpt",
            "Cpt_auto",
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

ALLOWED_MODULES = {"u", "math", "control", "solution"}

ALLOWED_CALLABLES = {
    f"{group['prefix']}{callable_name}": None
    for group in WHITELISTED_CALLABLES.values()
    for callable_name in group["callables"]  # type: ignore[union-attr]
}

EVALUATION_BLACKLIST = {
    f"{group['prefix']}{callable_name}": None
    for group in WHITELISTED_CALLABLES.values()
    for callable_name in group["callables"]  # type: ignore[union-attr]
    if not group["evaluate"]
}


# ---------------------------------------------------------------------------
# IMPORT_FUNCTIONS — lazy loaders for expression runtime modules
# ---------------------------------------------------------------------------


def _import_units(_: Any) -> Any:
    return unit_symbols


def _import_math(_: Any) -> Any:
    from flow360_schema.models.functions import math

    return math


def _import_control(_: Any) -> Any:
    from flow360_schema.models.variables import control

    return control


def _import_solution(_: Any) -> Any:
    from flow360_schema.models.variables import solution

    return solution


IMPORT_FUNCTIONS = {
    "u": _import_units,
    "math": _import_math,
    "control": _import_control,
    "solution": _import_solution,
}


# ---------------------------------------------------------------------------
# default_context singleton (Category 6: isolated for future removal)
# ---------------------------------------------------------------------------

default_context = EvaluationContext(
    CallableResolver(ALLOWED_CALLABLES, ALLOWED_MODULES, IMPORT_FUNCTIONS, EVALUATION_BLACKLIST)
)


# ---------------------------------------------------------------------------
# Solver internal variable names (from Python client utils.py, pure data)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Legacy output field names (from Python client output_fields.py AllFieldNames)
# These get a more specific error message than generic solver variable names.
# ---------------------------------------------------------------------------

LEGACY_OUTPUT_FIELD_NAMES = {
    "Cf",
    "CfVec",
    "Cp",
    "Cpt",
    "Cpt_auto",
    "Mach",
    "SpalartAllmaras_hybridModel",
    "T",
    "VelocityRelative",
    "betMetrics",
    "betMetricsPerDisk",
    "gradW",
    "heatFlux",
    "heatTransferCoefficientStaticTemperature",
    "heatTransferCoefficientTotalTemperature",
    "kOmega",
    "kOmegaSST_hybridModel",
    "linearResidualNavierStokes",
    "linearResidualTransition",
    "linearResidualTurbulence",
    "localCFL",
    "lowMachPreconditionerSensor",
    "mut",
    "mutRatio",
    "nodeForcesPerUnitArea",
    "nodeNormals",
    "nuHat",
    "numericalDissipationFactor",
    "pressure",
    "pressure_pa",
    "primitiveVars",
    "qcriterion",
    "residualHeatSolver",
    "residualNavierStokes",
    "residualTransition",
    "residualTurbulence",
    "s",
    "solutionNavierStokes",
    "solutionTransition",
    "solutionTurbulence",
    "velocity",
    "velocity_m_per_s",
    "velocity_magnitude",
    "velocity_magnitude_m_per_s",
    "velocity_x",
    "velocity_x_m_per_s",
    "velocity_y",
    "velocity_y_m_per_s",
    "velocity_z",
    "velocity_z_m_per_s",
    "vorticity",
    "vorticityMagnitude",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "wallDistance",
    "wallFunctionMetric",
    "wall_shear_stress_magnitude",
    "wall_shear_stress_magnitude_pa",
    "yPlus",
}


def clear_context() -> None:
    """Clear user variables from default_context, keep solver variables (names with '.')."""
    for name in list(default_context._values):
        if "." not in name:
            default_context._dependency_graph.remove_variable(name)
    default_context._values = {name: value for name, value in default_context._values.items() if "." in name}
