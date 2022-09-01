import os

from flow360 import Env
from flow360.component.surface_mesh import SurfaceMesh


def test_from_cloud():
    surface_mesh = SurfaceMesh(name="test")
