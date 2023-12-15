import json
import os
from typing import Literal, Optional, Type, Union

import pydantic as pd

import flow360 as fl
from flow360 import TimeStepping
from flow360.component.flow360_params.flow360_params import (
    ExpressionInitialCondition,
    FreestreamInitialCondition,
)
from flow360.component.flow360_params.params_base import Flow360BaseModel
from flow360.component.flow360_params.unit_system import TimeType
from flow360.component.types import PositiveInt


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(type_obj: Type[Flow360BaseModel], folder_name):
    data = type_obj.flow360_schema()
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


# pylint: disable=E0213
class _UnsteadyTimeStepping(TimeStepping):
    """
    Unsteady time stepping component
    """

    physical_steps: PositiveInt = pd.Field(alias="physicalSteps")
    max_pseudo_steps: Optional[PositiveInt] = pd.Field(alias="maxPseudoSteps")
    time_step_size: TimeType.Positive = pd.Field(
        alias="timeStepSize",
    )


# pylint: disable=E0213
class _SteadyTimeStepping(TimeStepping):
    """
    Steady time stepping component
    """

    physical_steps: Literal[1] = pd.Field(alias="physicalSteps", const=True)
    max_pseudo_steps: Optional[PositiveInt] = pd.Field(alias="maxPseudoSteps")
    time_step_size: Literal["inf"] = pd.Field(alias="timeStepSize", default="inf", const=True)


class _TimeSteppings(Flow360BaseModel):
    time_stepping: Union[_SteadyTimeStepping, _UnsteadyTimeStepping] = pd.Field(
        alias="timeStepping"
    )


class _FluidProperties(Flow360BaseModel):
    fluid_properties: Union[fl.AirDensityTemperature, fl.AirPressureTemperature] = pd.Field(
        alias="fluidProperties"
    )


class _InitialConditions(Flow360BaseModel):
    initial_conditions: Union[FreestreamInitialCondition, ExpressionInitialCondition] = pd.Field(
        alias="initialConditions"
    )


write_schemas(fl.NavierStokesSolver, "navier-stokes")
write_schemas(fl.Geometry, "geometry")
write_schemas(fl.SlidingInterface, "sliding-interface")
write_schemas(fl.TransitionModelSolver, "transition-model")
write_schemas(fl.HeatEquationSolver, "heat-equation")
write_schemas(fl.PorousMedium, "porous-media")
write_schemas(fl.ActuatorDisk, "actuator-disk")
write_schemas(fl.BETDisk, "bet-disk")
write_schemas(fl.SurfaceOutput, "surface-output")
write_schemas(fl.SliceOutput, "slice-output")
write_schemas(fl.VolumeOutput, "volume-output")
write_schemas(fl.AeroacousticOutput, "aeroacoustic-output")
write_schemas(fl.MonitorOutput, "monitor-output")
write_schemas(fl.IsoSurfaceOutput, "iso-surface-output")

write_schemas(_Freestreams, "freestream")
write_schemas(_TimeSteppings, "time-stepping")
write_schemas(_FluidProperties, "fluid-properties")
write_schemas(_TurbulenceModelSolvers, "turbulence-model")
write_schemas(_InitialConditions, "initial-conditions")

write_schemas(fl.Surfaces, "surfaces")
write_schemas(fl.VolumeZones, "volume-zones")
write_schemas(fl.Boundaries, "boundaries")
write_schemas(fl.Slices, "slices")
write_schemas(fl.IsoSurfaces, "iso-surfaces")
write_schemas(fl.Monitors, "monitors")
