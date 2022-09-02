import os

from flow360 import Env
from flow360.component.flow360_solver_params import Flow360MeshParams
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.volume_mesh import (
    VolumeMesh,
    VolumeMeshDownloadable,
    VolumeMeshLog,
)


def test_from_cloud():
    Env.dev.active()
    mesh = VolumeMesh.from_cloud("768aaf6b-37d9-471d-8246-9f0da9060bb6")
    assert mesh
    # mesh.download_log(to_file="./test_surface_mesh.log")
    # assert os.path.exists("./test_surface_mesh.log")
    # mesh.download_log()
    # assert os.path.exists(
    #     "./3f358de7-432e-4a1f-af26-ad53a3b84088/logs/flow360_surface_mesh.user.log"
    # )
    #
    # mesh.download_log(to_file="..")
    # assert os.path.exists(
    #     "../3f358de7-432e-4a1f-af26-ad53a3b84088/logs/flow360_surface_mesh.user.log"
    # )


def test_download_log():
    Env.dev.active()
    mesh = VolumeMesh.from_cloud("768aaf6b-37d9-471d-8246-9f0da9060bb6")
    mesh.download_log(VolumeMeshLog.USER_LOG, to_file="./test_volume_mesh.log")
    assert os.path.exists("./test_volume_mesh.log")


def test_download():
    Env.dev.active()
    mesh = VolumeMesh.from_cloud("768aaf6b-37d9-471d-8246-9f0da9060bb6")
    mesh.download(file_name="Flow360Mesh.json")
    assert os.path.exists("768aaf6b-37d9-471d-8246-9f0da9060bb6/Flow360Mesh.json")
    mesh.download(file_name=VolumeMeshDownloadable.CONFIG_JSON)


def test_create():
    Env.dev.active()
    mesh = SurfaceMesh.from_cloud("3f358de7-432e-4a1f-af26-ad53a3b84088")
    mesh = VolumeMesh.from_surface_mesh(
        "Volume Mesh", mesh.surface_mesh_id, "768aaf6b-37d9-471d-8246-9f0da9060bb6/Flow360Mesh.json"
    )
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


def test_ugrid_file():
    Env.dev.active()

    params = Flow360MeshParams.parse_raw(
        """
        {
        "boundaries": {
            "noSlipWalls": [
                "fluid/fuselage",
                "fluid/leftWing",
                "fluid/rightWing"
            ]
        }
    }
        """
    )
    mesh = VolumeMesh.from_ugrid_file(
        "test_ugrid", "3f358de7-432e-4a1f-af26-ad53a3b84088/geometry.csm", params
    )
    assert mesh
