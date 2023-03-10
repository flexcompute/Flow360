import pytest

from flow360.component.volume_mesh import VolumeMesh, VolumeMeshList

from flow360.log import set_logging_level

set_logging_level("DEBUG")

from .mock_server import mock_response


def test_volume_mesh_list(mock_response):
    list = VolumeMeshList()

    mesh = list[0]
    print(mesh.info)

    deleted = [item for item in list if item.info.deleted]

    assert len(list) == 100
    assert len(deleted) == 0
    assert mesh.status.value == "uploaded"
    assert mesh.status.is_final()

    for mesh in list:
        assert isinstance(mesh, VolumeMesh)
        assert mesh.status.is_final()


def test_volume_mesh_list_with_deleted(mock_response):
    list = VolumeMeshList(include_deleted=True)

    deleted = [item for item in list if item.info.deleted]
    assert len(deleted) == 16

    for mesh in list:
        assert isinstance(mesh, VolumeMesh)


def test_volume_mesh_list_with_deleted_without_limit(mock_response):
    list = VolumeMeshList(include_deleted=True, limit=None)

    deleted = [item for item in list if item.info.deleted]
    assert len(list) == 521
    assert len(deleted) == 194

    for mesh in list:
        assert isinstance(mesh, VolumeMesh)


def test_volume_mesh_list_with_limit(mock_response):
    list = VolumeMeshList(limit=10)

    deleted = [item for item in list if item.info.deleted]
    assert len(list) == 10
    assert len(deleted) == 0

    for mesh in list:
        assert isinstance(mesh, VolumeMesh)


def test_volume_mesh_list_without_limit(mock_response):
    list = VolumeMeshList(limit=None)

    deleted = [item for item in list if item.info.deleted]
    assert len(list) == 326
    assert len(deleted) == 0

    for mesh in list:
        assert isinstance(mesh, VolumeMesh)
