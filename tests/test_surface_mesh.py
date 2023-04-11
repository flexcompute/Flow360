import pytest
import unittest

from flow360.component.meshing.params import (
    SurfaceMeshingParams,
    Face,
)

from flow360 import exceptions as ex

from flow360.component.surface_mesh import SurfaceMesh

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_draft_surface_mesh():
    with pytest.raises(ex.ValueError):
        sm = SurfaceMesh.new("file.unsupported", params=None)

    with pytest.raises(ex.FileError):
        sm = SurfaceMesh.new("file_does_not_exist.csm", params=None)

    with pytest.raises(ex.ValueError):
        sm = SurfaceMesh.new("data/surface_mesh/test.csm", params=None)

    sm = SurfaceMesh.new(
        "data/surface_mesh/test.csm",
        params=SurfaceMeshingParams(
            max_edge_length=0.1, faces={"mysphere": Face(max_edge_length=0.05)}
        ),
    )
    assert sm
