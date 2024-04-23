import pytest

import flow360
from flow360.component.flow360_params.boundaries import NoSlipWall
from flow360.component.flow360_params.flow360_params import (
    Flow360MeshParams,
    Flow360Params,
    FreestreamFromMach,
    MeshBoundary,
)
from flow360.component.volume_mesh import (
    CompressionFormat,
    UGRIDEndianness,
    VolumeMesh,
    VolumeMeshFileFormat,
    VolumeMeshMeta,
    get_boundaries_from_file,
    get_no_slip_walls,
    validate_cgns,
)
from flow360.exceptions import Flow360RuntimeError, Flow360ValueError
from tests.data.volume_mesh_list import volume_mesh_list_raw

from .utils import compare_to_ref, to_file_from_file_test


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

    with flow360.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": NoSlipWall(),
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    to_file_from_file_test(param)
    to_file_from_file_test(param.boundaries)

    walls = get_no_slip_walls(param)
    assert walls
    assert len(walls) == 3


def test_validate_cgns():
    with flow360.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": NoSlipWall(),
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

        with pytest.raises(Flow360ValueError):
            validate_cgns(
                "data/volume_mesh/cylinder.cgns", param, solver_version="release-22.2.0.0"
            )

        param = Flow360Params(
            boundaries={
                "fluid/wall": NoSlipWall(),
                "fluid/farfield": NoSlipWall(),
                "fluid/periodic_0_l": NoSlipWall(),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
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
    with pytest.raises(Flow360RuntimeError):
        mesh_format = VolumeMeshFileFormat.detect(filename)

    file = "sdfdlkjd/kjsdf.ugrid"
    cmp, filename = CompressionFormat.detect(file)
    mesh_format = VolumeMeshFileFormat.detect(filename)
    with pytest.raises(Flow360RuntimeError):
        endianess = UGRIDEndianness.detect(filename)


def test_volume_mesh_list_with_incorrect_data():
    v = VolumeMeshMeta(**volume_mesh_list_raw[0])
    assert v.status.value == "uploaded"
    assert type(v.mesh_params) is Flow360MeshParams
    assert type(v.mesh_params.boundaries) is MeshBoundary
    assert v.mesh_params.boundaries.no_slip_walls == ["1", "wall"]

    v = VolumeMeshMeta(**volume_mesh_list_raw[1])
    assert v.status.value == "uploaded"
    assert type(v.mesh_params) is Flow360MeshParams
    assert type(v.mesh_params.boundaries) is MeshBoundary
    assert v.mesh_params.boundaries.no_slip_walls == ["4"]

    v = VolumeMeshMeta(**volume_mesh_list_raw[2])
    assert v.status.value == "uploaded"
    assert type(v.mesh_params) is Flow360MeshParams
    assert type(v.mesh_params.boundaries) is MeshBoundary
    assert v.mesh_params.boundaries.no_slip_walls == ["1"]

    item_incorrect1 = volume_mesh_list_raw[3]
    v = VolumeMeshMeta(**item_incorrect1)
    assert v.status.value == "error"
    assert v.mesh_params is None

    item_incorrect2 = volume_mesh_list_raw[4]
    v = VolumeMeshMeta(**item_incorrect2)
    assert v.status.value == "error"
    assert v.mesh_params is None


def test_volume_mesh_json():
    param = Flow360MeshParams("ref/flow360mesh/eg1.json")
    assert param.boundaries.no_slip_walls[0] == "1"

    for file in ["ref/flow360mesh/eg2.json", "ref/flow360mesh/eg3.json"]:
        param = Flow360MeshParams(file)
        compare_to_ref(param, file, content_only=True)


def test_volume_mesh():
    params = Flow360MeshParams(boundaries=MeshBoundary(no_slip_walls=[1]))
    VolumeMesh.from_file(file_name="data/cylinder.cgns", params=params)
