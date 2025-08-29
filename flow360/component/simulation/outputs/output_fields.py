"""
Output field definitions

This module defines the available output field names for Flow360 simulations,
including both standard non-dimensional fields and dimensioned fields in physical units.

It also provides support for dimensioned output fields, which automatically generate
UserDefinedField entries to output values in physical units rather than Flow360's
internal non-dimensional units.

Dimensioned field format:
    {base_field}_{component?}_{unit}

Where:
    - base_field: The base field name (velocity, pressure, temperature, etc.)
    - component: Optional component for vector fields (x, y, z, magnitude)
    - unit: The physical unit (m_per_s, pa, etc.)

Examples:
    - velocity_magnitude_m_per_s: Velocity magnitude in meters per second
    - velocity_x_m_per_s: X-component of velocity in meters per second
    - pressure_pa: Pressure in pascals
"""

from typing import List, Literal, get_args, get_origin

from flow360.component.simulation.conversion import (
    compute_udf_dimensionalization_factor,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    LiquidOperatingCondition,
)
from flow360.component.simulation.unit_system import u

# Coefficient of pressure
# Coefficient of total pressure
# Gradient of primitive solution
# k and omega
# Mach number
# Turbulent viscosity
# Turbulent viscosity and freestream dynamic viscosity ratio
# Spalart-Almaras variable
# rho, u, v, w, p (density, 3 velocities and pressure)
# Q criterion
# N-S residual
# Transition residual
# Turbulence residual
# Entropy
# N-S solution
# Transition solution
# Turbulence solution
# Temperature
# Velocity (non-dimensional)
# Velocity X component (non-dimensional)
# Velocity Y component (non-dimensional)
# Velocity Z component (non-dimensional)
# Velocity Magnitude (non-dimensional)
# Pressure (non-dimensional)
# Vorticity
# Vorticity Magnitude
# Wall distance
# NumericalDissipationFactor sensor
# Heat equation residual
# Velocity with respect to non-inertial frame
# Low-Mach preconditioner factor
# Velocity (dimensioned, m/s)
# Velocity X component (dimensioned, m/s)
# Velocity Y component (dimensioned, m/s)
# Velocity Z component (dimensioned, m/s)
# Velocity Magnitude (dimensioned, m/s)
# Pressure (dimensioned, Pa)
CommonFieldNames = Literal[
    "Cp",
    "Cpt",
    "gradW",
    "kOmega",
    "Mach",
    "mut",
    "mutRatio",
    "nuHat",
    "primitiveVars",
    "qcriterion",
    "residualNavierStokes",
    "residualTransition",
    "residualTurbulence",
    "s",
    "solutionNavierStokes",
    "solutionTransition",
    "solutionTurbulence",
    "T",
    "velocity",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "velocity_magnitude",
    "pressure",
    "vorticity",
    "vorticityMagnitude",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "wallDistance",
    "numericalDissipationFactor",
    "residualHeatSolver",
    "VelocityRelative",
    "lowMachPreconditionerSensor",
    # Include dimensioned fields here too
    "velocity_m_per_s",
    "velocity_x_m_per_s",
    "velocity_y_m_per_s",
    "velocity_z_m_per_s",
    "velocity_magnitude_m_per_s",
    "pressure_pa",
]

# Skin friction coefficient vector
# Magnitude of CfVec
# Non-dimensional heat flux
# Wall normals
# Spalart-Allmaras variable
# Non-dimensional wall distance
# Wall function metrics
# Surface heat transfer coefficient (static temperature as reference)
# Surface heat transfer coefficient (total temperature as reference)
# Wall shear stress magnitude (non-dimensional)
# Wall shear stress magnitude (dimensioned, Pa)
SurfaceFieldNames = Literal[
    CommonFieldNames,
    "CfVec",
    "Cf",
    "heatFlux",
    "nodeNormals",
    "nodeForcesPerUnitArea",
    "yPlus",
    "wallFunctionMetric",
    "heatTransferCoefficientStaticTemperature",
    "heatTransferCoefficientTotalTemperature",
    "wall_shear_stress_magnitude",
    "wall_shear_stress_magnitude_pa",
]

# BET Metrics
# BET Metrics per Disk
# Linear residual of Navier-Stokes solver
# Linear residual of turbulence solver
# Linear residual of transition solver
# Hybrid RANS-LES output for Spalart-Allmaras solver
# Hybrid RANS-LES output for kOmegaSST solver
# Local CFL number
VolumeFieldNames = Literal[
    CommonFieldNames,
    "betMetrics",
    "betMetricsPerDisk",
    "linearResidualNavierStokes",
    "linearResidualTurbulence",
    "linearResidualTransition",
    "SpalartAllmaras_hybridModel",
    "kOmegaSST_hybridModel",
    "localCFL",
]

SliceFieldNames = VolumeFieldNames

# Pressure
# Density
# Mach number
# Q criterion
# Entropy
# Temperature
# Coefficient of pressure
# Total pressure coefficient
# Turbulent viscosity
# Spalart-Almaras variable
# Vorticity magnitude
IsoSurfaceFieldNames = Literal[
    "Mach",
    "qcriterion",
    "s",
    "T",
    "Cp",
    "Cpt",
    "mut",
    "nuHat",
    "vorticityMagnitude",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "velocity_magnitude",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]

AllFieldNames = Literal[CommonFieldNames, SurfaceFieldNames, VolumeFieldNames, IsoSurfaceFieldNames]

InvalidOutputFieldsForLiquid = Literal[
    "residualNavierStokes",
    "residualTransition",
    "residualTurbulence",
    "solutionNavierStokes",
    "T",
    "Mach",
    "linearResidualNavierStokes",
    "linearResidualTurbulence",
    "linearResidualTransition",
    "SpalartAllmaras_DDES",
    "kOmegaSST_DDES",
    "heatFlux",
    "heatTransferCoefficientStaticTemperature",
    "heatTransferCoefficientTotalTemperature",
]
# pylint: disable=no-member
_FIELD_UNIT_MAPPING = {
    # Standard non-dimensioned fields - (unit, unit_system)
    "*": (None, "flow360"),
    # Dimensioned fields - (unit quantity, unit_system)
    "velocity_m_per_s": (u.m / u.s, "SI"),
    "velocity_magnitude_m_per_s": (u.m / u.s, "SI"),
    "velocity_x_m_per_s": (u.m / u.s, "SI"),
    "velocity_y_m_per_s": (u.m / u.s, "SI"),
    "velocity_z_m_per_s": (u.m / u.s, "SI"),
    "pressure_pa": (u.Pa, "SI"),
    "wall_shear_stress_magnitude_pa": (u.Pa, "SI"),
}


def get_unit_for_field(field_name: str):
    """
    Get the physical unit for a given field name.

    Parameters:
    -----------
    field_name : str
        The field name to get the unit for

    Returns:
    --------
    Tuple[Optional[Union[str, unyt.Unit]], str]
        A tuple containing (unit, unit_system) where:
        - unit: None for non-dimensioned fields, unyt.Unit for dimensioned fields
        - unit_system: "flow360" for non-dimensioned fields, "SI" for dimensioned fields
    """
    if field_name in _FIELD_UNIT_MAPPING:
        return _FIELD_UNIT_MAPPING[field_name]

    return _FIELD_UNIT_MAPPING["*"]


FIELD_TYPE_3DVECTOR = "3dvector"
FIELD_TYPE_SCALAR = "scalar"

_FIELD_TYPE_INFO = {
    "velocity": {
        "type": FIELD_TYPE_3DVECTOR,
    },
    "velocity_magnitude": {
        "type": FIELD_TYPE_SCALAR,
    },
    "velocity_x": {
        "type": FIELD_TYPE_SCALAR,
    },
    "velocity_y": {
        "type": FIELD_TYPE_SCALAR,
    },
    "velocity_z": {
        "type": FIELD_TYPE_SCALAR,
    },
    "pressure": {
        "type": FIELD_TYPE_SCALAR,
    },
    "vorticity_x": {
        "type": FIELD_TYPE_SCALAR,
    },
    "vorticity_y": {
        "type": FIELD_TYPE_SCALAR,
    },
    "vorticity_z": {
        "type": FIELD_TYPE_SCALAR,
    },
}

# Predefined UDF expressions
PREDEFINED_UDF_EXPRESSIONS = {
    "velocity": "velocity[0] = primitiveVars[1] * velocityScale;"
    + "velocity[1] = primitiveVars[2] * velocityScale;"
    + "velocity[2] = primitiveVars[3] * velocityScale;",
    "velocity_magnitude": "double velocity[3];"
    + "velocity[0] = primitiveVars[1];"
    + "velocity[1] = primitiveVars[2];"
    + "velocity[2] = primitiveVars[3];"
    + "velocity_magnitude = magnitude(velocity) * velocityScale;",
    "velocity_x": "velocity_x = primitiveVars[1] * velocityScale;",
    "velocity_y": "velocity_y = primitiveVars[2] * velocityScale;",
    "velocity_z": "velocity_z = primitiveVars[3] * velocityScale;",
    "pressure_": "double gamma = 1.4;pressure_ = (usingLiquidAsMaterial) ? "
    + "(primitiveVars[4] - 1.0 / gamma) * (velocityScale * velocityScale) : primitiveVars[4];",
    "wall_shear_stress_magnitude": "wall_shear_stress_magnitude = "
    + "magnitude(wallShearStress) * (velocityScale * velocityScale);",
    "vorticity_x": "vorticity_x = (gradPrimitive[3][1] - gradPrimitive[2][2]) * velocityScale;",
    "vorticity_y": "vorticity_y = (gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;",
    "vorticity_z": "vorticity_z = (gradPrimitive[2][0] - gradPrimitive[1][1]) * velocityScale;",
}


def _apply_vector_conversion(
    *, base_udf_expression: str, base_field: str, field_name: str, conversion_factor: float
):
    """Apply conversion for vector fields"""
    factor = 1.0 / conversion_factor
    return (
        f"double {base_field}[3];"
        f"{base_udf_expression}"
        f"{field_name}[0] = {base_field}[0] * {factor};"
        f"{field_name}[1] = {base_field}[1] * {factor};"
        f"{field_name}[2] = {base_field}[2] * {factor};"
    )


def _apply_scalar_conversion(
    *, base_udf_expression: str, base_field: str, field_name: str, conversion_factor: float
):
    """Apply conversion for scalar fields"""
    factor = 1.0 / conversion_factor
    return (
        f"double {base_field};" f"{base_udf_expression}" f"{field_name} = {base_field} * {factor};"
    )


def generate_predefined_udf(field_name, params):
    """
    Generate UserDefinedField expression for a dimensioned field.

    Parameters:
    -----------
    field_name : str
        Field name (e.g., 'velocity', 'velocity_m_per_s', 'pressure_pa', 'wall_shear_stress_magnitude_pa')
    params : SimulationParams
        The simulation parameters object for unit conversion

    Returns:
    --------
    str or None
        The expression for the UserDefinedField, or None if no matching base expression is found.
    """
    valid_field_names = get_field_values(AllFieldNames)
    if field_name not in valid_field_names:
        return None

    matching_keys = [key for key in PREDEFINED_UDF_EXPRESSIONS if field_name.startswith(key)]
    if not matching_keys:
        return None

    # Longer keys take precedence (e.g., "velocity_x" over "velocity")
    base_field = max(matching_keys, key=len)
    base_expr = PREDEFINED_UDF_EXPRESSIONS[base_field]

    unit, _ = get_unit_for_field(field_name)

    if unit is None:
        return base_expr

    coefficient, _ = compute_udf_dimensionalization_factor(
        params=params,
        requested_unit=unit,
        using_liquid_op=isinstance(params.operating_condition, LiquidOperatingCondition),
    )
    conversion_factor = 1.0 / coefficient

    field_info = _FIELD_TYPE_INFO.get(base_field, {"type": FIELD_TYPE_SCALAR})
    field_type = field_info["type"]

    if field_type == FIELD_TYPE_3DVECTOR:
        return _apply_vector_conversion(
            base_udf_expression=base_expr,
            base_field=base_field,
            field_name=field_name,
            conversion_factor=conversion_factor,
        )
    return _apply_scalar_conversion(
        base_udf_expression=base_expr,
        base_field=base_field,
        field_name=field_name,
        conversion_factor=conversion_factor,
    )


def _get_field_values(field_type, names):
    for arg in get_args(field_type):
        if get_origin(arg) is Literal:
            _get_field_values(arg, names)
        elif isinstance(arg, str):
            names += [arg]


def get_field_values(field_type) -> List[str]:
    """Retrieve field names from a nested literal type as list of strings"""
    values = []
    _get_field_values(field_type, values)
    return values


def append_component_to_output_fields(output_fields: List[str]) -> List[str]:
    """
    If "velocity" or "vorticity" is in the list, append their respective magnitude in output

    Parameters:
    -----------
    output_fields : List[str]
        The list of output fields to modify.

    Returns:
    --------
    List[str]
        The modified list of output fields with the component appended.
    """
    output_fields_with_component = []
    for field in output_fields:
        output_fields_with_component.append(field)
        if field == "velocity" and "velocity_magnitude" not in output_fields:
            output_fields_with_component.append("velocity_magnitude")
        if field == "vorticity" and "vorticityMagnitude" not in output_fields:
            output_fields_with_component.append("vorticityMagnitude")
    return output_fields_with_component
