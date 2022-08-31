from flow360 import Env
from flow360.component.surface_mesh import SurfaceMesh


def test_from_cloud():
    Env.dev.active()
    mesh = SurfaceMesh.from_cloud("c5a02ca8-4d9d-4bec-a53b-b3c40b7aa924")
    assert mesh
