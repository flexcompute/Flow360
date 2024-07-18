"""
Output field definitions
"""

from typing import List, Literal, get_args, get_origin

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
    "Velocity with respect to non-inertial frame",
    "Low-Mach preconditioner factor",
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
    "numericalDissipationFactor",
    "residualHeatSolver",
    "VelocityRelative",
    "lowMachPreconditionerSensor",
]

SurfaceFieldNamesFull = Literal[
    CommonFieldNamesFull,
    "Skin friction coefficient vector",
    "Magnitude of CfVec",
    "Non-dimensional heat flux",
    "Wall normals",
    "Spalart-Allmaras variable",
    "Non-dimensional wall distance",
    "Wall function metrics",
]

SurfaceFieldNames = Literal[
    CommonFieldNames,
    "CfVec",
    "Cf",
    "heatFlux",
    "nodeNormals",
    "nodeForcesPerUnitArea",
    "yPlus",
    "wallFunctionMetric",
]

VolumeFieldNamesFull = Literal[
    CommonFieldNamesFull,
    "BET Metrics",
    "BET Metrics per Disk",
    "SpalartAllmaras_DDES",
    "kOmegaSST_DDES",
]

SliceFieldNamesFull = VolumeFieldNamesFull

VolumeFieldNames = Literal[
    CommonFieldNames, "betMetrics", "betMetricsPerDisk", "SpalartAllmaras_DDES", "kOmegaSST_DDES"
]

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

AllFieldNamesFull = Literal[
    CommonFieldNamesFull, SurfaceFieldNamesFull, VolumeFieldNamesFull, IsoSurfaceFieldNamesFull
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


def get_aliases(name, raise_on_not_found=False) -> List[str]:
    """Retrieve all aliases for the given field full name or shorthand"""
    short = get_field_values(AllFieldNames)
    full = get_field_values(AllFieldNamesFull)

    if name in short:
        i = short.index(name)
        return [name, full[i]]

    if name in full:
        i = full.index(name)
        return [name, short[i]]

    if not raise_on_not_found:
        return [name, name]
    raise ValueError(f"{name} is not a valid output field name.")


def to_short(name, raise_on_not_found=False) -> str:
    """Retrieve shorthand equivalent of output field"""
    short = get_field_values(AllFieldNames)
    full = get_field_values(AllFieldNamesFull)

    if name in short:
        return name
    if name in full:
        i = full.index(name)
        return short[i]

    if not raise_on_not_found:
        return name
    raise ValueError(f"{name} is not a valid output field name.")


def to_full(name, raise_on_not_found=False) -> str:
    """Retrieve full name equivalent of output field"""
    short = get_field_values(AllFieldNames)
    full = get_field_values(AllFieldNamesFull)

    if name in full:
        return name
    if name in short:
        i = short.index(name)
        return full[i]

    if not raise_on_not_found:
        return name
    raise ValueError(f"{name} is not a valid output field name.")


if len(get_field_values(AllFieldNames)) != len(get_field_values(AllFieldNamesFull)):
    raise ImportError(
        "Full names and shorthands for output fields have mismatched lengths, which is not allowed"
    )


def _distribute_shared_output_fields(solver_values: dict, item_names: str):
    if "output_fields" not in solver_values or solver_values["output_fields"] is None:
        return
    shared_fields = solver_values.pop("output_fields")
    shared_fields = [to_short(field) for field in shared_fields]
    if solver_values[item_names] is not None:
        for name in solver_values[item_names].names():
            item = solver_values[item_names][name]
            for field in shared_fields:
                if item.output_fields is None:
                    item.output_fields = []
                if field not in item.output_fields:
                    item.output_fields.append(field)
