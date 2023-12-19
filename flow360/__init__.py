"""
This module is flow360.
"""

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
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    SupersonicInflow,
    WallFunction,
)
from .component.flow360_params.flow360_params import (
    ActuatorDisk,
    AeroacousticOutput,
    AirDensityTemperature,
    AirPressureTemperature,
    AnimationSettings,
    AnimationSettingsExtended,
    BETDisk,
    Boundaries,
    Flow360MeshParams,
    Flow360Params,
    FreestreamFromMach,
    FreestreamFromMachReynolds,
    FreestreamFromVelocity,
    Geometry,
    HeatEquationSolver,
    IncompressibleNavierStokesSolver,
    IsoSurfaceOutput,
    IsoSurfaces,
    KOmegaSST,
    MeshBoundary,
    MonitorOutput,
    Monitors,
    NavierStokesSolver,
    NoneSolver,
    PorousMedium,
    ProbeMonitor,
    SliceOutput,
    Slices,
    SlidingInterface,
    SpalartAllmaras,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    Surfaces,
    TransitionModelSolver,
    UnvalidatedFlow360Params,
    VolumeOutput,
    VolumeZones,
    ZeroFreestream,
    ZeroFreestreamFromVelocity,
    air,
)
from .component.flow360_params.time_stepping import (
    AdaptiveCFL,
    RampCFL,
    SteadyTimeStepping,
    UnsteadyTimeStepping,
)
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
    ReferenceFrame,
)
from .component.folder import Folder
from .component.meshing.params import SurfaceMeshingParams, VolumeMeshingParams
from .component.surface_mesh import SurfaceMesh
from .component.surface_mesh import SurfaceMeshList as MySurfaceMeshes
from .component.volume_mesh import VolumeMesh
from .component.volume_mesh import VolumeMeshList as MyVolumeMeshes
from .environment import Env
from .user_config import UserConfig
from .version import __version__
