"""
This module is flow360.
"""
from .cli import flow360
from .environment import Env
from .version import __version__

from .component.surface_mesh import SurfaceMesh
from .component.surface_mesh import SurfaceMeshList as MySurfaceMeshes
from .component.volume_mesh import VolumeMesh
from .component.volume_mesh import VolumeMeshList as MyVolumeMeshes
from .component.case import Case
from .component.case import CaseList as MyCases
from .component import flow360_params

from .component.meshing.params import SurfaceMeshingParams, VolumeMeshingParams
from .component import meshing

from .component.flow360_params import (
    Flow360MeshParams,
    MeshBoundary,
    Flow360Params,
    Boundaries,
    Geometry,
    Freestream,
    TimeStepping,
    TimeSteppingCFL,
    TurbulenceModelSolver,
    NavierStokesSolver,
    SlidingInterface,
)

from .component.flow360_params import (
    NoSlipWall,
    SlipWall,
    FreestreamBoundary,
    IsothermalWall,
    SubsonicOutflowPressure,
    SubsonicOutflowMach,
    SubsonicInflow,
    SlidingInterfaceBoundary,
    WallFunction,
    MassInflow,
    MassOutflow,
)


from .cloud.s3_utils import ProgressCallbackInterface


# pylint: disable=too-few-public-methods,invalid-name
class turbulence:
    """turbulece models shortcut: eg. flow360.turbulence.SA"""

    SA = flow360_params.TurbulenceModelModelType.SA
    SST = flow360_params.TurbulenceModelModelType.SST
    NONE = flow360_params.TurbulenceModelModelType.NONE
