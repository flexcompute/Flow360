import os

from flow360 import Env
from flow360.component.surface_mesh import SurfaceMesh


def test_from_cloud():
    Env.dev.active()
    mesh = SurfaceMesh.from_cloud("3f358de7-432e-4a1f-af26-ad53a3b84088")
    assert mesh
    mesh.download_log(to_file="./test_surface_mesh.log")
    assert os.path.exists("./test_surface_mesh.log")
    mesh.download_log()
    assert os.path.exists(
        "./3f358de7-432e-4a1f-af26-ad53a3b84088/logs/flow360_surface_mesh.user.log"
    )

    mesh.download_log(to_file="..")
    assert os.path.exists(
        "../3f358de7-432e-4a1f-af26-ad53a3b84088/logs/flow360_surface_mesh.user.log"
    )


def test_download():
    Env.dev.active()
    mesh = SurfaceMesh.from_cloud("3f358de7-432e-4a1f-af26-ad53a3b84088")
    mesh.download(file_name="geometry.csm")


def test_create_update():
    Env.dev.active()
    mesh = SurfaceMesh.from_file("test_name", "3f358de7-432e-4a1f-af26-ad53a3b84088/geometry.csm")
    assert mesh
    mesh.submit()


def test_from_geometry():
    Env.dev.active()
    mesh = SurfaceMesh.from_cloud("3f358de7-432e-4a1f-af26-ad53a3b84088")
    mesh.download("config.json")

    mesh = SurfaceMesh.from_geometry(
        "test_name",
        "3f358de7-432e-4a1f-af26-ad53a3b84088/geometry.csm",
        "3f358de7-432e-4a1f-af26-ad53a3b84088/config.json",
    )
    mesh.submit()
