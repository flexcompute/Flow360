"""
Output field definitions
"""

from typing import List, Literal, get_args, get_origin

# Coefficient of pressure
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
# Vorticity
# Wall distance
# NumericalDissipationFactor sensor
# Heat equation residual
# Velocity with respect to non-inertial frame
# Low-Mach preconditioner factor
CommonFieldNames = Literal[
    "Cp",
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
    "vorticity",
    "wallDistance",
    "numericalDissipationFactor",
    "residualHeatSolver",
    "VelocityRelative",
    "lowMachPreconditionerSensor",
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
]

# BET Metrics
# BET Metrics per Disk
# Coefficient of total pressure
# Linear residual of Navier-Stokes solver
# Linear residual of turbulence solver
# Linear residual of transition solver
# DDES output for Spalart-Allmaras solver
# DDES output for kOmegaSST solver
# Local CFL number
VolumeFieldNames = Literal[
    CommonFieldNames,
    "betMetrics",
    "betMetricsPerDisk",
    "Cpt",
    "linearResidualNavierStokes",
    "linearResidualTurbulence",
    "linearResidualTransition",
    "SpalartAllmaras_DDES",
    "kOmegaSST_DDES",
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
# Turbulent viscosity
# Spalart-Almaras variable
IsoSurfaceFieldNames = Literal[
    "p",
    "rho",
    "Mach",
    "qcriterion",
    "s",
    "T",
    "Cp",
    "mut",
    "nuHat",
]

AllFieldNames = Literal[CommonFieldNames, SurfaceFieldNames, VolumeFieldNames, IsoSurfaceFieldNames]


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


def _distribute_shared_output_fields(solver_values: dict, item_names: str):
    if "output_fields" not in solver_values or solver_values["output_fields"] is None:
        return
    shared_fields = solver_values.pop("output_fields")
    if solver_values[item_names] is not None:
        for name in solver_values[item_names].names():
            item = solver_values[item_names][name]
            for field in shared_fields:
                if item.output_fields is None:
                    item.output_fields = []
                if field not in item.output_fields:
                    item.output_fields.append(field)
