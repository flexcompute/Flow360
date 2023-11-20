import pytest

from flow360 import (
    Case,
    Flow360Params,
    FreestreamFromVelocity,
    Geometry,
    SI_unit_system,
    VolumeMesh,
    air,
)
from flow360.exceptions import RuntimeError, ValueError
from flow360.log import set_logging_level

set_logging_level("DEBUG")

from .mock_server import mock_response
from .utils import mock_id


def test_case(mock_response):
    with SI_unit_system:
        case = Case.create(
            name="hi",
            params=Flow360Params(
                geometry=Geometry(mesh_unit="m"),
                freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                fluid_properties=air,
            ),
            volume_mesh_id=mock_id,
        )
    case_2 = case.copy()
    case_3 = case.retry()
    case_4 = case.fork()
    case_5 = case.continuation()
    print(case_5)
    case.submit()


def test_retry_with_parent(mock_response):
    with SI_unit_system:
        case = Case.create(
            name="hi",
            params=Flow360Params(
                geometry=Geometry(mesh_unit="m"),
                freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                fluid_properties=air
            ),
            volume_mesh_id=mock_id,
        )
    case2 = case.continuation()
    case3 = case2.copy(name="case-parent-copy")

    case.submit()
    case2.submit()
    case3.submit()


def test_fork_from_draft(mock_response):
    with SI_unit_system:
        case = Case.create(
            name="hi",
            params=Flow360Params(
                geometry=Geometry(mesh_unit="m"),
                freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                fluid_properties=air
            ),
            volume_mesh_id=mock_id,
        )
    case2 = case.continuation()
    with pytest.raises(RuntimeError):
        case2.submit()


def test_parent_id(mock_response):
    vm = VolumeMesh(mock_id)
    with SI_unit_system:
        case = vm.create_case(
            name="hi",
            params=Flow360Params(
                geometry=Geometry(mesh_unit="m"),
                freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                fluid_properties=air
            ),
        )
    print(case)
    case.submit()


    with SI_unit_system:
        case = Case.create(
            name="hi",
            params=Flow360Params(
                geometry=Geometry(mesh_unit="m"),
                freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                fluid_properties=air
            ),
            parent_id=mock_id,
        )
    print(case)
    case.submit()

    with pytest.raises(ValueError):
        with SI_unit_system:
            case = Case.create(
                name="hi",
                params=Flow360Params(
                    geometry=Geometry(mesh_unit="m"),
                    freestream=FreestreamFromVelocity(velocity=286, alpha=3.06),
                    fluid_properties=air
                ),
                parent_id="incorrect parentId",
            )
        print(case)
        case.submit()
