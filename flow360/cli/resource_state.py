"""
Shared resource state helpers for the Flow360 CLI.
"""

from __future__ import annotations

import time

import click

from flow360.cli.resource_refs import ResourceRefError, parse_resource_ref
from flow360.component.resource_status import (
    is_final_resource_status,
    is_success_resource_status,
)


class WaitTimeoutError(RuntimeError):
    """Raised when a wait loop exceeds the requested timeout."""

    def __init__(self, state):
        super().__init__("Timed out while waiting for terminal resource state.")
        self.state = state


def serialize_resource_state(info, *, default_type=None):
    """Project a resource info payload into the CLI lifecycle-state contract."""
    status = info.get("status")
    resource_type = info.get("type") or default_type
    return {
        "id": info.get("id"),
        "type": resource_type,
        "status": status,
        "is_terminal": is_final_resource_status(resource_type, status),
        "is_success": is_success_resource_status(resource_type, status),
        "updated_at": info.get("updatedAt"),
    }


def get_resource_state_for_type(resource_type, resource_id):
    """Fetch and serialize lifecycle state for a known resource type."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import get_resource_webapi_class

    try:
        webapi_cls = get_resource_webapi_class(resource_type)
    except ValueError as error:
        raise click.ClickException(
            f"Lifecycle state for {resource_type} resources is not supported."
        ) from error

    info = webapi_cls(resource_id).get_info()
    payload = serialize_resource_state(info, default_type=resource_type)
    if resource_type == "Case":
        payload["mesh_id"] = info.get("caseMeshId") or info.get("meshId") or info.get("parentId")
    return payload


def get_resource_state(resource_id):
    """Fetch lifecycle state for a typed Flow360 resource id."""
    try:
        resource_ref = parse_resource_ref(resource_id)
    except ResourceRefError as error:
        raise click.ClickException(str(error)) from error

    return get_resource_state_for_type(resource_ref.resource_type, resource_ref.id)


def wait_for_resource_state(resource_id, *, timeout, poll_interval):
    """Poll a resource until it reaches a terminal state or times out."""
    deadline = time.monotonic() + timeout
    last_state = None

    while True:
        last_state = get_resource_state(resource_id)
        if last_state["is_terminal"]:
            return last_state
        if time.monotonic() >= deadline:
            raise WaitTimeoutError(last_state)
        time.sleep(poll_interval)
