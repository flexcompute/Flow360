import os
from sys import exc_info

import pytest

from flow360.component.flow360_solver_params import (
    Flow360MeshParams,
    Flow360Params,
    MeshBoundary,
    NoSlipWall,
    SlidingInterface,
)
from flow360.component.volume_mesh import (
    get_boundaries_from_file,
    get_boundries_from_sliding_interfaces,
    get_no_slip_walls,
    validate_cgns,
    VolumeMeshFileFormat,
    CompressionFormat,
    UGRIDEndianness,
)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_get_boundary_names():
    names = get_boundaries_from_file("data/volume_mesh/cylinder.cgns", "release-22.2.0.0")
    assert len(names) == 4

    names = get_boundaries_from_file("data/cylinder.cgns", "release-22.2.1.0")
    assert len(names) == 4


def test_get_no_slip_walls():
    param = Flow360MeshParams(
        boundaries=MeshBoundary.parse_obj(
            {"noSlipWalls": ["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]}
        )
    )
    assert param

    walls = get_no_slip_walls(param)
    assert walls
    assert len(walls) == 3

    param = Flow360Params(
        boundaries={
            "fluid/fuselage": NoSlipWall(),
            "fluid/leftWing": NoSlipWall(),
            "fluid/rightWing": NoSlipWall(),
        }
    )
    walls = get_no_slip_walls(param)
    assert walls
    assert len(walls) == 3


def test_get_walls_from_sliding_interfaces():
    param = Flow360Params(
        sliding_interfaces=SlidingInterface.parse_obj(
            {"stationaryPatches": ["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]}
        )
    )
    walls = get_boundries_from_sliding_interfaces(param)
    assert walls
    assert len(walls) == 3


def test_validate_cgns():
    param = Flow360Params(
        boundaries={
            "fluid/fuselage": NoSlipWall(),
            "fluid/leftWing": NoSlipWall(),
            "fluid/rightWing": NoSlipWall(),
        }
    )

    with pytest.raises(ValueError):
        validate_cgns("data/volume_mesh/cylinder.cgns", param, solver_version="release-22.2.0.0")

    param = Flow360Params(
        boundaries={
            "fluid/wall": NoSlipWall(),
            "fluid/farfield": NoSlipWall(),
            "fluid/periodic_0_l": NoSlipWall(),
        }
    )

    validate_cgns("data/volume_mesh/cylinder.cgns", param, solver_version="release-22.2.0.0")
    validate_cgns("data/cylinder.cgns", param)


def test_mesh_filename_detection():
    files_correct = [
        ("sdfdlkjd/kjsdf.lb8.ugrid.gz", ".lb8.ugrid.gz"),
        ("sdfdlkjd/kjsdf.b8.ugrid.gz", ".b8.ugrid.gz"),
        ("sdfdlkjd/kjsdf.lb8.ugrid.bz2", ".lb8.ugrid.bz2"),
        ("sdfdlkjd/kjsdf.lb8.ugrid", ".lb8.ugrid"),
        ("sdfdlkjd/kjsdf.lb8.cgns", ".cgns"),
        ("sdfdlkjd/kjsdf.cgns", ".cgns"),
        ("sdfdlkjd/kjsdf.cgns.gz", ".cgns.gz"),
        ("sdfdlkjd/kjsdf.cgns.bz2", ".cgns.bz2"),
    ]

    for file, expected in files_correct:
        cmp, filename = CompressionFormat.detect(file)
        mesh_format = VolumeMeshFileFormat.detect(filename)
        endianess = UGRIDEndianness.detect(filename)
        assert expected == f"{endianess.ext()}{mesh_format.ext()}{cmp.ext()}"

    file = "sdfdlkjd/kjsdf.cgns.ad"
    cmp, filename = CompressionFormat.detect(file)
    with pytest.raises(RuntimeError):
        mesh_format = VolumeMeshFileFormat.detect(filename)

    file = "sdfdlkjd/kjsdf.ugrid"
    cmp, filename = CompressionFormat.detect(file)
    mesh_format = VolumeMeshFileFormat.detect(filename)
    with pytest.raises(RuntimeError):
        endianess = UGRIDEndianness.detect(filename)
