"""
Contains classes that we put under the volumes key of Simulation constructor.
"""

from abc import ABCMeta
from typing import Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.material import Material
from flow360.component.simulation.operating_condition import OperatingConditionTypes
from flow360.component.simulation.physics_components import (
    ActuatorDisk,
    BETDisk,
    HeatEquationSolver,
    NavierStokesSolver,
    PorousMediumBox,
    TransitionModelSolver,
    TurbulenceModelSolverType,
)
from flow360.component.simulation.zones import BoxZone, CylindricalZone

##:: Physical Volume ::##


class VolumeBase(Flow360BaseModel): ...


class PhysicalVolumeBase(Flow360BaseModel, metaclass=ABCMeta):
    entities: list[Union[BoxZone | CylindricalZone]]
    operating_condition: OperatingConditionTypes = pd.Field()
    material: Optional[Material] = pd.Field()


class FluidDynamics(PhysicalVolumeBase):
    # Contains all the common fields every fluid dynamics zone should have
    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field()
    turbulence_model_solver: Optional[TurbulenceModelSolverType] = pd.Field()
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field()
    ...


class ActuatorDisk(FluidDynamics):
    ## or inherit from (Flow360BaseModel) so it is consistent with our solver capability (no specification on per zone basis)
    actuator_disks: ActuatorDisk = pd.Field()


class BETDisk(FluidDynamics):
    bet_disks: BETDisk = pd.Field()


class Rotation(FluidDynamics):
    # Needs dedicated implementation as importing an existing zone class here is inconsistent.
    rmp: float = pd.Field()
    ...


class MovingReferenceFrame(FluidDynamics):
    # Needs dedicated implementation as importing an existing zone class here is inconsistent.
    ...


class PorousMedium(FluidDynamics):
    porous_media: PorousMediumBox = pd.Field()


class SolidHeatTransfer(PhysicalVolumeBase):
    heat_equation_solver: HeatEquationSolver = pd.Field()


VolumeTypes = Union[
    FluidDynamics,
    ActuatorDisk,
    BETDisk,
    Rotation,
    MovingReferenceFrame,
    PorousMedium,
    SolidHeatTransfer,
]
