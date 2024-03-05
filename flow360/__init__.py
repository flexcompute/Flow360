"""
This module is flow360.
"""

import os

from numpy import pi

from . import global_exception_handler, units
from .accounts_utils import Accounts
from .cli import flow360
from .cloud.s3_utils import ProgressCallbackInterface
from .component import meshing
from .component.case import Case
from .component.case import CaseList as MyCases
from .component.flow360_params import solvers
from .component.flow360_params.boundaries import (
    FreestreamBoundary,
    HeatFluxWall,
    IsothermalWall,
    MassInflow,
    MassOutflow,
    NoSlipWall,
    PressureOutflow,
    RiemannInvariant,
    RotationallyPeriodic,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    SupersonicInflow,
    SymmetryPlane,
    TranslationallyPeriodic,
    VelocityInflow,
    WallFunction,
)
from .component.flow360_params.flow360_output import Slice, Slices
from .component.flow360_params.flow360_params import (
    ActuatorDisk,
    AeroacousticOutput,
    AirDensityTemperature,
    AirPressureTemperature,
    AnimationSettings,
    AnimationSettingsExtended,
    BETDisk,
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
    Boundaries,
    Flow360MeshParams,
    Flow360Params,
    ForcePerArea,
    FreestreamFromMach,
    FreestreamFromMachReynolds,
    FreestreamFromVelocity,
    Geometry,
    HeatEquationSolver,
    IsoSurfaceOutput,
    IsoSurfaces,
    KOmegaSST,
    MeshBoundary,
    MonitorOutput,
    Monitors,
    NoneSolver,
    PorousMedium,
    ProbeMonitor,
    SliceOutput,
    SlidingInterface,
    SpalartAllmaras,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    Surfaces,
    TransitionModelSolver,
    UnvalidatedFlow360Params,
    UserDefinedDynamic,
    VolumeOutput,
    VolumeZones,
    ZeroFreestream,
    ZeroFreestreamFromVelocity,
    air,
)
from .component.flow360_params.initial_condition import (
    ExpressionInitialCondition,
    FreestreamInitialCondition,
)
from .component.flow360_params.solvers import (
    IncompressibleNavierStokesSolver,
    LinearSolver,
    NavierStokesSolver,
    PressureCorrectionSolver,
)
from .component.flow360_params.time_stepping import (
    AdaptiveCFL,
    RampCFL,
    SteadyTimeStepping,
    TimeStepping,
    UnsteadyTimeStepping,
)
from .component.flow360_params.turbulence_quantities import TurbulenceQuantities
from .component.flow360_params.unit_system import (
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    flow360_unit_system,
    imperial_unit_system,
)
from .component.flow360_params.volume_zones import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    ReferenceFrame,
    ReferenceFrameDynamic,
    ReferenceFrameExpression,
    ReferenceFrameOmegaDegrees,
    ReferenceFrameOmegaRadians,
)
from .component.folder import Folder
from .component.meshing.params import (
    Aniso,
    BoxRefinement,
    CylinderRefinement,
    Edges,
    Face,
    Faces,
    Farfield,
    ProjectAniso,
    RotorDisk,
    SurfaceMeshingParams,
    UseAdjacent,
    Volume,
    VolumeMeshingParams,
)
from .component.surface_mesh import SurfaceMesh
from .component.surface_mesh import SurfaceMeshList as MySurfaceMeshes
from .component.volume_mesh import VolumeMesh
from .component.volume_mesh import VolumeMeshList as MyVolumeMeshes
from .environment import Env
from .flags import Flags
from .user_config import UserConfig
from .version import __version__
