"""Relay import for meshing volume zone models."""

from flow360_schema.models.entities.volume_entities import CustomVolume
from flow360_schema.models.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CentralBelt,
    CustomZones,
    FullyMovingFloor,
    MeshSliceOutput,
    RotationCylinder,
    RotationSphere,
    RotationVolume,
    StaticFloor,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
    WheelBelts,
    WindTunnelFarfield,
    _FarfieldAllowingEnclosedEntities,
)
