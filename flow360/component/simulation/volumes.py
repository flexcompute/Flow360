"""
Contains classes that we put under the volumes key of Simulation constructor.
"""

from typing import Optional, Tuple, Union

import pydantic as pd
from material import Material
from operating_condition import OperatingConditionTypes
from zones import BoxZone, CylindricalZone
from abc import ABCMeta

from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    BETDisk,
    PorousMediumBox,
)
from flow360.component.flow360_params.params_base import Flow360BaseModel
from flow360.component.flow360_params.solvers import (
    HeatEquationSolver,
    TurbulenceModelSolverType,
)
from flow360.component.simulation.physics_components import (
    NavierStokesSolver,
    TransitionModelSolver,
)

##:: Physical Volume ::##

class VolumeBase(Flow360BaseModel):
    


class PhysicalVolumeBase(Flow360BaseModel, metaclass = ABCMeta):
    entities: list[Union[BoxZone | CylindricalZone]]
    operating_condition: OperatingConditionTypes = pd.Field()
    material: Optional[Material] = pd.Field()


class FluidDynamics(PhysicalVolumeBase):
    # Contains all the common fields every fluid dynamics zone should have
    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field()
    turbulence_model_solver: Optional[TurbulenceModelSolverType] = pd.Field(
        discriminator="model_type"
    )
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
