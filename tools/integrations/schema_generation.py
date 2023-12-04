import json
import os
from typing import Literal, Optional, Type, Union

import pydantic as pd

import flow360 as fl
from flow360 import TimeStepping
from flow360.component.flow360_params.params_base import Flow360BaseModel
from flow360.component.flow360_params.unit_system import TimeType
from flow360.component.types import PositiveInt


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(type_obj: Type[Flow360BaseModel]):
    data = type_obj.flow360_schema()
    schema = json.dumps(data, indent=2)
    write_to_file(f"./data/{type_obj.__name__}.json", schema)
    ui_schema = json.dumps(type_obj.flow360_ui_schema(), indent=2)
    if ui_schema is not None:
        write_to_file(f"./data/{type_obj.__name__}.ui.json", ui_schema)


if not os.path.exists(f"./data/"):
    os.mkdir(f"./data/")


class Freestreams(Flow360BaseModel):
    freestream: Union[
        fl.FreestreamFromVelocity,
        fl.FreestreamFromMach,
        fl.ZeroFreestreamFromVelocity,
        fl.ZeroFreestream,
    ] = pd.Field()


# pylint: disable=E0213
class UnsteadyTimeStepping(TimeStepping):
    """
    Unsteady Ttme stepping component
    """

    physical_steps: PositiveInt = pd.Field(alias="physicalSteps")
    max_pseudo_steps: Optional[PositiveInt] = pd.Field(alias="maxPseudoSteps")
    time_step_size: TimeType.Positive = pd.Field(
        alias="timeStepSize",
    )


# pylint: disable=E0213
class SteadyTimeStepping(TimeStepping):
    """
    Steady time stepping component
    """

    physical_steps: Literal[1] = pd.Field(alias="physicalSteps", const=True)
    max_pseudo_steps: Optional[PositiveInt] = pd.Field(alias="maxPseudoSteps")
    time_step_size: Literal["inf"] = pd.Field(alias="timeStepSize", default="inf", const=True)


class TimeSteppings(Flow360BaseModel):
    time_stepping: Union[SteadyTimeStepping, UnsteadyTimeStepping] = pd.Field(alias="timeStepping")


write_schemas(fl.NavierStokesSolver)
write_schemas(fl.Geometry)
write_schemas(Freestreams)
write_schemas(fl.SlidingInterface)
write_schemas(fl.TurbulenceModelSolverSA)
write_schemas(fl.TurbulenceModelSolverSST)
write_schemas(fl.TransitionModelSolver)
write_schemas(fl.HeatEquationSolver)
write_schemas(fl.NoneSolver)
write_schemas(fl.PorousMedium)
write_schemas(TimeSteppings)
write_schemas(fl.ActuatorDisk)
write_schemas(fl.BETDisk)
write_schemas(fl.SurfaceOutput)
write_schemas(fl.SliceOutput)
write_schemas(fl.VolumeOutput)
write_schemas(fl.AeroacousticOutput)
write_schemas(fl.MonitorOutput)
write_schemas(fl.IsoSurfaceOutput)

write_schemas(fl.Surfaces)
write_schemas(fl.VolumeZones)
write_schemas(fl.Boundaries)
write_schemas(fl.Slices)
write_schemas(fl.IsoSurfaces)
write_schemas(fl.Monitors)
