import json
import os
from typing import Literal, Optional, Type, Union

import pydantic as pd

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    ExpressionInitialCondition,
    FreestreamInitialCondition,
    TimeStepping,
)
from flow360.component.flow360_params.params_base import Flow360BaseModel


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(
    type_obj: Type[Flow360BaseModel], folder_name, root_property=None, swap_fields=None
):
    data = type_obj.flow360_schema()
    if root_property is not None:
        current = data
        for item in root_property:
            current = current[item]
        data[root_property[-1]] = current
        del data["properties"]
        del data["required"]
    if swap_fields is not None:
        for key, value in swap_fields.items():
            data["properties"][key] = value
    schema = json.dumps(data, indent=2)
    name = type_obj.__name__
    if name.startswith("_"):
        name = name[1:]
    if not os.path.exists(f"./data/{folder_name}"):
        os.mkdir(f"./data/{folder_name}")
    write_to_file(f"./data/{folder_name}/json-schema.json", schema)
    ui_schema = json.dumps(type_obj.flow360_ui_schema(), indent=2)
    if ui_schema is not None:
        write_to_file(f"./data/{folder_name}/ui-schema.json", ui_schema)


if not os.path.exists(f"./data/"):
    os.mkdir(f"./data/")


class _Freestreams(Flow360BaseModel):
    """
    Freestreams wrapper for schema generation
    """

    freestream: Union[
        fl.FreestreamFromVelocity,
        fl.FreestreamFromMach,
        fl.ZeroFreestreamFromVelocity,
        fl.ZeroFreestream,
        fl.FreestreamFromMachReynolds,
    ] = pd.Field()


class _TurbulenceModelSolvers(Flow360BaseModel):
    """
    Turbulence solvers wrapper for schema generation
    """

    solver: Union[fl.SpalartAllmaras, fl.KOmegaSST, fl.NoneSolver]


class _TimeSteppings(Flow360BaseModel):
    time_stepping: TimeStepping = pd.Field(alias="timeStepping", options=["Steady", "Unsteady"])


class _FluidProperties(Flow360BaseModel):
    fluid_properties: Union[fl.AirDensityTemperature, fl.AirPressureTemperature] = pd.Field(
        alias="fluidProperties",
        options=["From density and temperature", "From pressure and temperature"],
    )


class _InitialConditions(Flow360BaseModel):
    initial_conditions: Union[FreestreamInitialCondition, ExpressionInitialCondition] = pd.Field(
        alias="initialConditions", options=["Freestream", "Expression"]
    )


write_schemas(fl.NavierStokesSolver, "navier-stokes")
write_schemas(fl.Geometry, "geometry")
write_schemas(fl.SlidingInterface, "sliding-interface")
write_schemas(fl.TransitionModelSolver, "transition-model")
write_schemas(fl.HeatEquationSolver, "heat-equation")
write_schemas(fl.PorousMedium, "porous-media")
write_schemas(fl.ActuatorDisk, "actuator-disk")
write_schemas(fl.BETDisk, "bet-disk")
write_schemas(fl.VolumeOutput, "volume-output")
write_schemas(fl.AeroacousticOutput, "aeroacoustic-output")
write_schemas(fl.SliceOutput, "slice-output", swap_fields={"slices": fl.Slices.flow360_schema()})
write_schemas(
    fl.MonitorOutput, "monitor-output", swap_fields={"monitors": fl.Monitors.flow360_schema()}
)
write_schemas(
    fl.SurfaceOutput, "surface-output", swap_fields={"surfaces": fl.Surfaces.flow360_schema()}
)
write_schemas(
    fl.IsoSurfaceOutput,
    "iso-surface-output",
    swap_fields={"isoSurfaces": fl.IsoSurfaces.flow360_schema()},
)

write_schemas(_Freestreams, "freestream", root_property=["properties", "freestream", "anyOf"])
write_schemas(
    _TimeSteppings, "time-stepping", root_property=["properties", "timeStepping", "anyOf"]
)
write_schemas(
    _FluidProperties, "fluid-properties", root_property=["properties", "fluidProperties", "anyOf"]
)
write_schemas(
    _TurbulenceModelSolvers, "turbulence-model", root_property=["properties", "solver", "anyOf"]
)
write_schemas(
    _InitialConditions,
    "initial-conditions",
    root_property=["properties", "initialConditions", "anyOf"],
)

write_schemas(fl.VolumeZones, "volume-zones")
write_schemas(fl.Boundaries, "boundaries")
