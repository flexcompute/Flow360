import unittest

import pytest

from flow360 import exceptions as ex
from flow360.component.meshing.params import Face, SurfaceMeshingParams
from flow360.component.surface_mesh import SurfaceMesh, SurfaceMeshFileFormat

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_draft_surface_mesh_create():
    with pytest.raises(ex.Flow360ValueError):
        sm = SurfaceMesh.create("file.unsupported", params=None)

    with pytest.raises(ex.Flow360FileError):
        sm = SurfaceMesh.create("file_does_not_exist.csm", params=None)

    with pytest.raises(ex.Flow360ValueError):
        sm = SurfaceMesh.create("data/surface_mesh/test.csm", params=None)

    sm = SurfaceMesh.create(
        "data/surface_mesh/test.csm",
        params=SurfaceMeshingParams(
            max_edge_length=0.1, faces={"mysphere": Face(max_edge_length=0.05)}
        ),
    )
    assert sm


def test_draft_surface_mesh_from_file():
    with pytest.raises(ex.Flow360ValueError):
        sm = SurfaceMesh.from_file("file.unsupported")

    with pytest.raises(ex.Flow360FileError):
        sm = SurfaceMesh.from_file("file_does_not_exist.stl")

    sm = SurfaceMesh.from_file("data/surface_mesh/airplaneGeometry.stl")
    assert sm


def test_mesh_filename_detection():
    files_correct = [
        ("sdfdlkjd/kjsdf.lb8.ugrid", ".lb8.ugrid"),
        ("sdfdlkjd/kjsdf.stl", ".stl"),
    ]
    for file, expected in files_correct:
        mesh_format = SurfaceMeshFileFormat.detect(file)
        assert expected == mesh_format
