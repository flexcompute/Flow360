"""
This module is flow360.
"""
from .cli import flow360
from .environment import Env

__version__ = "0.1.0"

from .component.volume_mesh import VolumeMesh
from .component.case import Case
from .component.flow360_solver_params import Flow360MeshParams, MeshBoundary, Flow360Params
