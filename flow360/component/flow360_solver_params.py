"""
Flow360 solver parameters
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import math


from pydantic import BaseModel, Extra, Field, validator


class MeshBoundary(BaseModel):
    """
    Mesh boundary
    """

    no_slip_walls: Union[List[str], List[int]] = Field(alias="noSlipWalls")

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True


class Boundary(BaseModel):
    """
    Basic Boundary class
    """

    type: str
    name: Optional[str]


class NoSlipWall(Boundary):
    """
    No slip wall boundary
    """

    type = "NoSlipWall"
    velocity: Optional[Union[float, str]] = Field(alias="Velocity")


class SlipWall(Boundary):
    """
    Slip wall boundary
    """

    type = "SlipWall"


class FreestreamBoundary(Boundary):
    """
    Freestream boundary
    """

    type = "Freestream"


BoundaryType = Union[NoSlipWall, SlipWall, FreestreamBoundary]


class ActuatorDisk(BaseModel):
    """
    Actuator disk component
    """

    center: Any
    axis_thrust: Any = Field(alias="axisThrust")
    thickness: Any
    force_per_area: Any = Field(alias="forcePerArea")


class SlidingInterface(BaseModel):
    """
    Sliding interface component
    """

    stationary_patches: Optional[List[str]] = Field(alias="stationaryPatches")
    rotating_patches: Optional[List[str]] = Field(alias="rotatingPatches")
    axis_of_rotation: Optional[List[int]] = Field(alias="axisOfRotation")
    center_of_rotation: Optional[List[int]] = Field(alias="centerOfRotation")


class TimeSteppingCFL(BaseModel):
    """
    CFL for time stepping component
    """

    initial: Optional[int] = Field(default=5)
    final: Optional[int] = Field(default=200)
    ramp_steps: Optional[int] = Field(alias="rampSteps", default=40)


# pylint: disable=E0213
class TimeStepping(BaseModel):
    """
    Time stepping component
    """

    physical_steps: Optional[int] = Field(alias="physicalSteps")
    max_pseudo_steps: Optional[int] = Field(alias="maxPseudoSteps")
    time_step_size: Optional[Union[float, str]] = Field(alias="timeStepSize", default="inf")
    CFL: Optional[TimeSteppingCFL] = Field(default=TimeSteppingCFL())

    @validator("time_step_size")
    def check_time_step_size(cls, value):
        """
        validates time step size
        """
        err_message = 'Time step size must be "inf" or positive number.'
        if isinstance(value, str):
            if value == "inf":
                return value
        elif isinstance(value, float):
            if math.isinf(value):
                return "inf"
            if value > 0:
                return value
        raise ValueError(err_message)

    def json(self, *args, **kwargs):
        """
        to json
        """
        return super().json(*args, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs)

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow
        validate_assignment = True


class Geometry(BaseModel):
    """
    Geometry component
    """

    ref_area: Optional[float] = Field(alias="refArea", default=1)
    moment_center: Optional[List[float]] = Field(alias="momentCenter", default=[0, 0, 0])
    moment_length: List[float] = Field(alias="momentLength")

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True


class Freestream(BaseModel):
    """
    Freestream component
    """

    Reynolds: Optional[float] = Field()
    Mach: float = Field()
    MachRef: Optional[float] = Field()
    mu_ref: Optional[float] = Field(alias="muRef")
    Temperature: float
    alpha: float = Field(alias="alphaAngle", default=0)
    beta: float = Field(alias="betaAngle", default=0)

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True


class NavierStokesSolver(BaseModel, extra=Extra.allow):
    """
    NavierStokesSolver component
    """

    absolute_tolerance: Optional[float] = Field(alias="absoluteTolerance", default=1e-10)
    kappaMUSCL: Optional[float] = Field(default=-1.0)


class TurbulenceModelModelType(str, Enum):
    """
    Turbulence model type
    """

    SA = "SpalartAllmaras"
    SST = "kOmegaSST"


class TurbulenceModelSolver(BaseModel):
    """
    TurbulenceModelSolver component
    """

    model_type: TurbulenceModelModelType = Field(
        alias="modelType", default=TurbulenceModelModelType.SA
    )
    absolute_tolerance: Optional[float] = Field(alias="absoluteTolerance", default=1e-10)
    kappaMUSCL: Optional[float] = Field(default=-1.0)

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow


class Flow360Params(BaseModel):
    """
    Flow360 solver parameters
    """

    geometry: Optional[Geometry]
    boundaries: Optional[Dict[str, BoundaryType]]
    time_stepping: Optional[TimeStepping] = Field(alias="timeStepping")
    sliding_interfaces: Optional[SlidingInterface] = Field(alias="slidingInterfaces")
    navier_stokes_solver: NavierStokesSolver = Field(
        alias="navierStokesSolver", default=NavierStokesSolver()
    )
    turbulence_model_solver: TurbulenceModelSolver = Field(
        alias="turbulenceModelSolver", default=TurbulenceModelSolver()
    )
    freestream: Optional[Freestream] = Field()
    bet_disks: Optional[List] = Field(alias="betDisks")
    actuator_disks: Optional[List] = Field(alias="actuatorDisks")

    def json(self, *args, **kwargs):
        """
        to json
        """
        return super().json(*args, by_alias=True, exclude_none=True, **kwargs)

    @classmethod
    def from_file(cls, file: str) -> Flow360Params:
        """
        read model from json file
        """
        return cls.parse_file(file)

    @classmethod
    def default(cls, steady: bool = True, solver_version: str = None) -> Flow360Params:
        """
        return default case settings
        """
        raise NotImplementedError
        # return cls.from_file("case.default.json")

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow


class Flow360MeshParams(BaseModel):
    """
    Flow360 mesh parameters
    """

    boundaries: MeshBoundary = Field()
    sliding_interfaces: Optional[List[SlidingInterface]] = Field(alias="slidingInterfaces")

    def json(self, *args, **kwargs):
        """
        to json
        """
        return super().json(*args, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs)

    @classmethod
    def from_file(cls, file: str) -> Flow360MeshParams:
        """
        read model from json file
        """
        return cls.parse_file(file)

    # pylint: disable=C0115,R0903
    class Config:
        allow_population_by_field_name = True
