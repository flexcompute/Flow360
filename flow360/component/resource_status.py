"""
Shared Flow360 resource status semantics.
"""

from __future__ import annotations

_DEFAULT_FINAL_STATUSES = frozenset(
    {
        "completed",
        "diverged",
        "error",
        "failed",
        "uploaded",
        "processed",
        "deleted",
    }
)

_FINAL_STATUSES_BY_RESOURCE_TYPE = {
    None: _DEFAULT_FINAL_STATUSES,
    "case": _DEFAULT_FINAL_STATUSES,
    "draft": _DEFAULT_FINAL_STATUSES,
    "geometry": frozenset({"error", "processed", "deleted"}),
    "surfacemesh": frozenset({"error", "processed", "deleted", "completed"}),
    "volumemesh": frozenset({"completed", "error"}),
}

_DEFAULT_SUCCESS_STATUSES = frozenset({"completed", "processed", "uploaded"})

_SUCCESS_STATUSES_BY_RESOURCE_TYPE = {
    None: _DEFAULT_SUCCESS_STATUSES,
    "case": frozenset({"completed", "processed"}),
    "draft": frozenset({"completed", "processed"}),
    "geometry": frozenset({"processed"}),
    "surfacemesh": frozenset({"processed", "completed"}),
    "volumemesh": frozenset({"completed"}),
}


def _normalize_status(status) -> str | None:
    if status is None:
        return None
    status = getattr(status, "value", status)
    if isinstance(status, str):
        return status.lower()
    return str(status).lower()


def _normalize_resource_type(resource_type) -> str | None:
    if resource_type is None:
        return None
    resource_type = getattr(resource_type, "value", resource_type)
    if not isinstance(resource_type, str):
        resource_type = str(resource_type)
    return resource_type.replace("-", "").replace("_", "").lower()


def is_final_resource_status(resource_type, status) -> bool:
    """Return whether a resource status should stop polling."""
    normalized_status = _normalize_status(status)
    if normalized_status is None:
        return False
    normalized_resource_type = _normalize_resource_type(resource_type)
    final_statuses = _FINAL_STATUSES_BY_RESOURCE_TYPE.get(
        normalized_resource_type, _DEFAULT_FINAL_STATUSES
    )
    return normalized_status in final_statuses


def is_success_resource_status(resource_type, status) -> bool:
    """Return whether a resource status represents a successful terminal state."""
    normalized_status = _normalize_status(status)
    if normalized_status is None:
        return False
    normalized_resource_type = _normalize_resource_type(resource_type)
    success_statuses = _SUCCESS_STATUSES_BY_RESOURCE_TYPE.get(
        normalized_resource_type, _DEFAULT_SUCCESS_STATUSES
    )
    return normalized_status in success_statuses
