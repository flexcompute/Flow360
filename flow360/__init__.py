"""
This module is flow360.
"""
from .cli import flow360
from .cloud.s3_utils import ProgressCallbackInterface
from .component import flow360_params, meshing
from .component.case import Case
from .component.case import CaseList as MyCases
from .component.flow360_params import (
    Boundaries,
    Flow360MeshParams,
    Flow360Params,
    Freestream,
    FreestreamBoundary,
    Geometry,
    IsothermalWall,
    MassInflow,
    MassOutflow,
    MeshBoundary,
    NavierStokesSolver,
    NoSlipWall,
    SlidingInterface,
    SlidingInterfaceBoundary,
    SlipWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    TimeStepping,
    TimeSteppingCFL,
    TurbulenceModelSolver,
    WallFunction,
)
from .component.meshing.params import SurfaceMeshingParams, VolumeMeshingParams
from .component.surface_mesh import SurfaceMesh
from .component.surface_mesh import SurfaceMeshList as MySurfaceMeshes
from .component.volume_mesh import VolumeMesh
from .component.volume_mesh import VolumeMeshList as MyVolumeMeshes
from .environment import Env
from .version import __version__


# pylint: disable=too-few-public-methods,invalid-name
class turbulence:
    """turbulece models shortcut: eg. flow360.turbulence.SA"""

    SA = flow360_params.TurbulenceModelModelType.SA
    SST = flow360_params.TurbulenceModelModelType.SST
    NONE = flow360_params.TurbulenceModelModelType.NONE
