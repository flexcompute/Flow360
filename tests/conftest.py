import os
from pathlib import Path

from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.validation.validation_context import (
    CASE,
    ParamsValidationInfo,
    ValidationContext,
)
from flow360.component.surface_mesh_v2 import SurfaceMeshMetaV2, SurfaceMeshV2
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)

import tempfile

import pytest

from flow360.file_path import flow360_dir
from flow360.log import set_logging_file, toggle_rotation

"""
Before running all tests redirect all test logging to a temporary log file, turn off log rotation
due to multi-threaded rotation being unsupported at this time
"""

pytest_plugins = ["tests.utils", "tests.mock_server"]


def pytest_configure():
    fo = tempfile.NamedTemporaryFile()
    fo.close()  # Windows workaround for shared files
    pytest.tmp_log_file = fo.name
    pytest.log_test_file = os.path.join(flow360_dir, "logs", "flow360_log_test.log")
    if os.path.exists(pytest.log_test_file):
        os.remove(pytest.log_test_file)
    set_logging_file(fo.name, level="DEBUG")
    toggle_rotation(False)


@pytest.fixture
def before_log_test(request):
    set_logging_file(pytest.log_test_file, level="DEBUG")


@pytest.fixture
def after_log_test():
    yield
    set_logging_file(pytest.tmp_log_file, level="DEBUG")


@pytest.fixture
def mock_validation_context():
    return ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )


@pytest.fixture
def mock_case_validation_context():
    return ValidationContext(
        levels=CASE, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )


@pytest.fixture
def mock_geometry():
    data_root = Path(__file__).parent / "data"
    geometry_meta = local_metadata_builder(
        id="geo-entity-provider",
        name="three-boxes",
        cloud_path_prefix="--",
        status="processed",
    )

    geometry = Geometry.from_local_storage(
        geometry_id=geometry_meta["id"],
        local_storage_path=data_root / geometry_meta["id"],
        meta_data=GeometryMeta(**geometry_meta),
    )
    return geometry


@pytest.fixture
def mock_surface_mesh():
    surface_data = Path(__file__).parent / "simulation" / "service" / "data"
    surface_meta = local_metadata_builder(
        id="surface-mesh",
        name="surface-mesh",
        cloud_path_prefix="--",
        status="processed",
    )

    surface_mesh = SurfaceMeshV2.from_local_storage(
        local_storage_path=surface_data,
        meta_data=SurfaceMeshMetaV2(**surface_meta),
    )
    assert surface_mesh._entity_info.type_name == "SurfaceMeshEntityInfo"
    return surface_mesh


@pytest.fixture
def mock_volume_mesh():
    data_root = Path(__file__).parent / "data"

    volume_meta = local_metadata_builder(
        id="vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
        name="VolumeMesh_v1",
        cloud_path_prefix="--",
    )
    volume_mesh = VolumeMeshV2.from_local_storage(
        local_storage_path=data_root / volume_meta["id"],
        meta_data=VolumeMeshMetaV2(**volume_meta),
    )
    return volume_mesh
