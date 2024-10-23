from io import StringIO

import pytest

from flow360 import Accounts
from flow360.cli.dict_utils import merge_overwrite
from flow360.component.utils import (
    CompressionFormat,
    MeshFileFormat,
    MeshNameParser,
    UGRIDEndianness,
    is_valid_uuid,
    shared_account_confirm_proceed,
    validate_type,
)
from flow360.component.volume_mesh import VolumeMeshMeta
from flow360.exceptions import Flow360TypeError, Flow360ValueError


def test_validate_type():
    validate_type("str", "meta", str)
    with pytest.raises(Flow360TypeError):
        validate_type("str", "meta", VolumeMeshMeta)


def test_merge_overwrite():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}

    dict1 = merge_overwrite(dict1, dict2)

    assert dict1 == {"a": 1, "b": 2, "c": 3, "d": 4}

    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}

    dict1 = merge_overwrite(dict1, dict2)

    assert dict1 == {"a": 1, "b": 3, "c": 4}

    dict1 = {"a": {"aa": 1, "ab": 2}}
    dict2 = {"a": {"ab": 3, "ac": 4}}

    dict1 = merge_overwrite(dict1, dict2)

    assert dict1 == {"a": {"aa": 1, "ab": 3, "ac": 4}}


def test_shared_confirm_proceed(mock_response, monkeypatch):
    Accounts.choose_shared_account("user1@test.com")

    monkeypatch.setattr("sys.stdin", StringIO("n\n"))

    assert not shared_account_confirm_proceed()

    assert not Accounts.shared_account_submit_is_confirmed()

    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    assert shared_account_confirm_proceed()

    assert Accounts.shared_account_submit_is_confirmed()


def test_valid_uuid():
    is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")
    is_valid_uuid("folder-123e4567-e89b-12d3-a456-426614174000")
    is_valid_uuid("g-123e4567-e89b-12d3-a456-426614174000")
    is_valid_uuid("long-length-prefix-123e4567-e89b-12d3-a456-426614174000")

    with pytest.raises(Flow360ValueError):
        is_valid_uuid("too-short-uuid")

    with pytest.raises(Flow360ValueError):
        is_valid_uuid(None)

    is_valid_uuid(None, allow_none=True)


def test_mesh_name_parser_uncompressed_ugrid():
    parser = MeshNameParser("testMesh.lb8.ugrid")
    assert parser.file_name_no_compression == "testMesh.lb8.ugrid"
    assert parser.endianness == UGRIDEndianness.LITTLE
    assert parser.format == MeshFileFormat.UGRID
    assert parser.compression == CompressionFormat.NONE
    assert parser.is_ugrid()
    assert not parser.is_compressed()


def test_mesh_name_parser_ascii_ugrid():
    parser = MeshNameParser("testMesh.ugrid")
    assert parser.file_name_no_compression == "testMesh.ugrid"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.UGRID
    assert parser.compression == CompressionFormat.NONE
    assert parser.is_ugrid()
    assert not parser.is_compressed()


def test_mesh_name_parser_ascii_compressed_ugrid():
    parser = MeshNameParser("testMesh.ugrid.zst")
    assert parser.file_name_no_compression == "testMesh.ugrid"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.UGRID
    assert parser.compression == CompressionFormat.ZST
    assert parser.is_ugrid()
    assert parser.is_compressed()


def test_mesh_name_parser_compressed_ugrid():
    parser = MeshNameParser("testMesh.lb8.ugrid.bz2")
    assert parser.file_name_no_compression == "testMesh.lb8.ugrid"
    assert parser.endianness == UGRIDEndianness.LITTLE
    assert parser.format == MeshFileFormat.UGRID
    assert parser.compression == CompressionFormat.BZ2
    assert parser.is_ugrid()
    assert parser.is_compressed()


def test_mesh_name_parser_compressed_cgns():
    parser = MeshNameParser("testMesh.cgns.bz2")
    assert parser.file_name_no_compression == "testMesh.cgns"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.CGNS
    assert parser.compression == CompressionFormat.BZ2
    assert parser.is_ugrid() == False
    assert parser.is_compressed()


def test_mesh_name_parser_uncompressed_cgns():
    parser = MeshNameParser("testMesh.cgns")
    assert parser.file_name_no_compression == "testMesh.cgns"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.CGNS
    assert parser.compression == CompressionFormat.NONE
    assert parser.is_ugrid() == False
    assert not parser.is_compressed()


def test_mesh_name_parser_stl():
    parser = MeshNameParser("testMesh.stl")
    assert parser.file_name_no_compression == "testMesh.stl"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.STL
    assert parser.compression == CompressionFormat.NONE
    assert parser.is_ugrid() == False
    assert not parser.is_compressed()


def test_mesh_name_parser_compressed_stl():
    parser = MeshNameParser("testMesh.stl.bz2")
    assert parser.file_name_no_compression == "testMesh.stl"
    assert parser.endianness == UGRIDEndianness.NONE
    assert parser.format == MeshFileFormat.STL
    assert parser.compression == CompressionFormat.BZ2
    assert parser.is_ugrid() == False
    assert parser.is_valid_surface_mesh()
    assert not parser.is_valid_volume_mesh()
    assert parser.is_compressed()


def test_mesh_name_parser_compressed_targz():
    parser = MeshNameParser("testMesh.lb8.ugrid.tar.gz")
    assert parser.file_name_no_compression == "testMesh.lb8.ugrid"
    assert parser.endianness == UGRIDEndianness.LITTLE
    assert parser.format == MeshFileFormat.UGRID
    assert parser.compression == CompressionFormat.TARGZ
    assert parser.is_ugrid() == True
    assert parser.is_valid_surface_mesh()
    assert parser.is_valid_volume_mesh()
    assert parser.is_compressed()
