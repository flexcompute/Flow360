"""
Output field definitions
"""
from typing import Literal, get_args, get_origin

CommonFieldNamesFull = Literal[
    "Coefficient of pressure",
    "Gradient of primitive solution",
    "k and omega",
    "Mach number",
    "Turbulent viscosity",
    "Turbulent viscosity and freestream dynamic viscosity ratio",
    "Spalart-Almaras variable",
    "rho, u, v, w, p (density, 3 velocities and pressure)",
    "Q criterion",
    "N-S residual",
    "Transition residual",
    "Turbulence residual",
    "Entropy",
    "N-S solution",
    "Transition solution",
    "Turbulence solution",
    "Temperature",
    "Vorticity",
    "Wall distance",
    "NumericalDissipationFactor sensor",
    "Heat equation residual",
]

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
    "lowNumericalDissipationSensor",
    "residualHeatSolver",
]

SurfaceFieldNamesFull = Literal[
    CommonFieldNamesFull,
    "Viscous stress coefficient vector",
    "Magnitude of CfVec",
    "Magnitude of CfVec normal to the wall",
    "Magnitude of CfVec tangent to the wall",
    "Non-dimensional heat flux",
    "Wall normals",
    "Spalart-Almaras variable",
    "Velocity in rotating frame",
    "Non-dimensional wall distance",
]

SurfaceFieldNames = Literal[
    CommonFieldNames,
    "CfVec",
    "Cf",
    "CfNormal",
    "CfTangent",
    "heatFlux",
    "nodeNormals",
    "nodeForcesPerUnitArea",
    "VelocityRelative",
    "yPlus",
    "wallFunctionMetric",
]

VolumeFieldNamesFull = CommonFieldNamesFull

SliceFieldNamesFull = VolumeFieldNamesFull

VolumeFieldNames = Literal[CommonFieldNames, "betMetrics", "betMetricsPerDisk"]

SliceFieldNames = VolumeFieldNames

IsoSurfaceFieldNamesFull = Literal[
    CommonFieldNamesFull,
    "Pressure",
    "Density",
    "Mach number",
    "Q criterion",
    "Entropy",
    "Temperature",
    "Coefficient of pressure",
    "Turbulent viscosity",
    "Spalart-Almaras variable",
]

IsoSurfaceFieldNames = Literal[
    CommonFieldNames,
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


def _get_field_values(field_type, names):
    for arg in get_args(field_type):
        if get_origin(arg) is Literal:
            _get_field_values(arg, names)
        elif isinstance(arg, str):
            names += [arg]


def get_field_values(field_type):
    """Retrieve field names from a nested literal type as list of strings"""
    values = []
    _get_field_values(field_type, values)
    return values
