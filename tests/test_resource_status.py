from flow360.component.resource_base import Flow360Status
from flow360.component.resource_status import (
    is_final_resource_status,
    is_success_resource_status,
)


def test_flow360_status_is_final_uses_shared_helper():
    assert Flow360Status.DIVERGED.is_final()
    assert Flow360Status.UPLOADED.is_final()
    assert not Flow360Status.RUNNING.is_final()


def test_resource_specific_final_status_semantics():
    assert is_final_resource_status("Case", "DIVERGED")
    assert is_final_resource_status("Geometry", "processed")
    assert is_final_resource_status("SurfaceMesh", "completed")
    assert is_final_resource_status("VolumeMesh", "completed")
    assert not is_final_resource_status("Geometry", "failed")
    assert not is_final_resource_status("SurfaceMesh", "failed")
    assert not is_final_resource_status("VolumeMesh", "failed")
    assert not is_final_resource_status("VolumeMesh", "uploaded")


def test_resource_specific_success_status_semantics():
    assert is_success_resource_status("Geometry", "processed")
    assert is_success_resource_status("SurfaceMesh", "completed")
    assert is_success_resource_status("VolumeMesh", "completed")
    assert not is_success_resource_status("Case", "diverged")
    assert not is_success_resource_status("VolumeMesh", "uploaded")
