"""
Contains classes that we put under the volumes key of Simulation constructor.
"""

from abc import ABCMeta
from typing import Optional, Tuple, Union

import pydantic as pd

import flow360.component.simulation.physics_components as components
from flow360.component.simulation.entities_base import EntitiesBase
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.material.material import Material
from flow360.component.simulation.operating_condition import OperatingConditionTypes
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.component.simulation.references import ReferenceGeometry


class Volume(Flow360BaseModel):
    mesh_volume_name: str = pd.Field()


##:: Physical Volume ::##


class PhysicalVolumeBase(EntitiesBase, metaclass=ABCMeta):
    operating_condition: OperatingConditionTypes = pd.Field()
    material: Optional[Material] = pd.Field()
    reference_geometry: Optional[ReferenceGeometry] = pd.Field()


class FluidDynamics(PhysicalVolumeBase):
    # Contains all the common fields every fluid dynamics zone should have
    # Note: Compute Reynolds from material and OperatingCondition
    navier_stokes_solver: Optional[components.NavierStokesSolver] = pd.Field()
    turbulence_model_solver: Optional[components.TurbulenceModelSolverType] = pd.Field()
    transition_model_solver: Optional[components.TransitionModelSolver] = pd.Field()


class ActuatorDisk(PhysicalVolumeBase):
    """Same as Flow360Param ActuatorDisks.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    pass


class BETDisk(PhysicalVolumeBase):
    """Same as Flow360Param BETDisk.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    pass


class Rotation(PhysicalVolumeBase):
    """Similar to Flow360Param ReferenceFrame.
    Note that `center`, `axis`, `radius`, `thickness` can be acquired from `entity` so they are not required anymore.
    Note: Should use the unit system to convert degree or degree per second to radian and radian per second
    """

    # (AKA omega) In conflict with `rotation_per_second`.
    angular_velocity: Union[float, pd.StrictStr] = pd.Field()
    # (AKA theta) Must be expression otherwise zone become static.
    rotation_angle_radians: pd.StrictStr = pd.Field()
    # unprescribed rotation is governed by UDD and expression will not be allowed.
    prescribed_rotation: bool = pd.Field()
    parent_entity: EntitiesBase = pd.Field()
    pass


class PorousMedium(FluidDynamics):
    """Same as Flow360Param PorousMediumBase."""

    pass


class SolidHeatTransfer(PhysicalVolumeBase):
    heat_equation_solver: components.HeatEquationSolver = pd.Field()


VolumeTypes = Union[
    FluidDynamics,
    ActuatorDisk,
    BETDisk,
    Rotation,
    PorousMedium,
    SolidHeatTransfer,
]
