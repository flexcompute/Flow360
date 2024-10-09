"""
This module is flow360 for simulation based models
"""

import os

from numpy import pi

from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation import services
from flow360.component.simulation import units as u
from flow360.component.simulation.meshing_param.params import MeshingParams, MeshingDefaults
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement, HeightBasedRefinement
from flow360.component.simulation.primitives import Surface, Edge, ReferenceGeometry
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.utils import _model_attribute_unlock
from flow360.component.simulation.operating_condition.operating_condition import AerospaceCondition, ThermalState
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady, RampCFL, AdaptiveCFL
from flow360.component.simulation.models.surface_models import Wall, Freestream, SlipWall, Outflow, Inflow, Periodic, SymmetryPlane
from flow360.component.simulation.models.volume_models import Fluid, Solid, ActuatorDisk, BETDisk, Rotation, PorousMedium
from flow360.component.simulation.outputs.outputs import SurfaceOutput, TimeAverageSurfaceOutput, VolumeOutput, TimeAverageVolumeOutput, SliceOutput, IsosurfaceOutput, SurfaceIntegralOutput, ProbeOutput, SurfaceProbeOutput, AeroAcousticOutput
from flow360.component.simulation.models.solver_numerics import (
    HeatEquationSolver,
    NavierStokesSolver,
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolverType,
    TurbulenceModelSolverType,
    LinearSolver,
)

__all__ = [
        "u",
        "SimulationParams",
        "SI_unit_system",
        "services",
        "MeshingParams",
        "MeshingDefaults",
        "SurfaceRefinement",
        "AutomatedFarfield",
        "SurfaceEdgeRefinement",
        "HeightBasedRefinement",
        "Surface",
        "Edge",
        "ReferenceGeometry",
        "AssetCache",
        "GeometryEntityInfo",
        "_model_attribute_unlock",
        "AerospaceCondition",
        "ThermalState",
        "Steady",
        "Unsteady",
        "RampCFL",
        "AdaptiveCFL",
        "Wall",
        "Freestream",
        "SlipWall",
        "Outflow",
        "Inflow",
        "Periodic",
        "SymmetryPlane",
        "Fluid",
        "Solid",
        "ActuatorDisk",
        "BETDisk",
        "Rotation",
        "PorousMedium",
        "SurfaceOutput",
        "TimeAverageSurfaceOutput",
        "VolumeOutput",
        "TimeAverageVolumeOutput",
        "SliceOutput",
        "IsosurfaceOutput",
        "SurfaceIntegralOutput",
        "ProbeOutput",
        "SurfaceProbeOutput",
        "AeroAcousticOutput",
        "HeatEquationSolver",
        "NavierStokesSolver",
        "NoneSolver",
        "SpalartAllmaras",
        "TransitionModelSolverType",
        "TurbulenceModelSolverType",
        "LinearSolver",

]
