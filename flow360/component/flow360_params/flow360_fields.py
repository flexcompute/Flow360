"""
Output field definitions
"""
from typing import List, Tuple, Union

_common_field_definitions = [
    ("Cp", "Coefficient of pressure"),
    ("gradW", "Gradient of primitive solution"),
    ("kOmega", "k and omega"),
    ("Mach", "Mach number"),
    ("mut", "Turbulent viscosity"),
    ("mutRatio", "Turbulent viscosity and freestream dynamic viscosity ratio"),
    ("nuHat", "Spalart-Almaras variable"),
    ("primitiveVars", "rho, u, v, w, p (density, 3 velocities and pressure)"),
    ("qcriterion", "Q criterion"),
    ("residualNavierStokes", "N-S residual"),
    ("residualTransition", "Transition residual"),
    ("residualTurbulence", "Turbulence residual"),
    ("s", "Entropy"),
    ("solutionNavierStokes", "N-S solution"),
    ("solutionTransition", "Transition solution"),
    ("solutionTurbulence", "Turbulence solution"),
    ("T", "Temperature"),
    ("vorticity", "Vorticity"),
    ("wallDistance", "Wall distance"),
    ("lowNumericalDissipationSensor", "NumericalDissipationFactor sensor"),
    ("residualHeatSolver", "Heat equation residual"),
]

_surface_field_definitions = [
    ("CfVec", "Viscous stress coefficient vector"),
    ("Cf", "Magnitude of CfVec"),
    ("CfNormal", "Magnitude of CfVec normal to the wall"),
    ("CfTangent", "Magnitude of CfVec tangent to the wall"),
    ("heatFlux", "Non-dimensional heat flux"),
    ("nodeNormals", "Wall normals"),
    ("nodeForcesPerUnitArea", "Spalart-Almaras variable"),
    ("VelocityRelative", "Velocity in rotating frame"),
    ("yPlus", "Non-dimensional wall distance"),
    ("wallFunctionMetric", None),
]

_volume_slice_field_definitions = [("betMetrics", None), ("betMetricsPerDisk", None)]

_isosurface_field_definitions = [
    ("p", "Pressure"),
    ("rho", "Density"),
    ("Mach", "Mach number"),
    ("qcriterion", "Q criterion"),
    ("s", "Entropy"),
    ("T", "Temperature"),
    ("Cp", "Coefficient of pressure"),
    ("mut", "Turbulent viscosity"),
    ("nuHat", "Spalart-Almaras variable"),
]


def _field_names(definitions: List[Tuple[str, Union[str, None]]], short=True):
    total = [field for field in (entry[1] for entry in definitions) if field is not None]
    if short:
        total += [field for field in (entry[0] for entry in definitions) if field is not None]
    return total


def output_names(types, short=True):
    field_list = []
    if "common" in types:
        field_list += _field_names(_common_field_definitions, short)
    if "surface" in types:
        field_list += _field_names(_surface_field_definitions, short)
    if "slice" in types or "volume" in types:
        field_list += _field_names(_volume_slice_field_definitions, short)
    if "iso_surface" in types:
        field_list += _field_names(_isosurface_field_definitions, short)
    return field_list
