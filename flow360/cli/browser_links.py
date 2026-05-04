"""Shared browser-link helpers for Flow360 CLI resources."""

from __future__ import annotations

import webbrowser
from urllib.parse import urlencode

from flow360.cli.resource_refs import ResourceRefError, parse_resource_ref
from flow360.environment import Env


def _is_root_folder_id(resource_id: str) -> bool:
    return resource_id == "ROOT.FLOW360" or resource_id.startswith("ROOT.FLOW360.")


def _get_project_scoped_resource_info(resource_type: str, resource_id: str) -> dict:
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import (
        CaseWebApi,
        DraftWebApi,
        GeometryWebApi,
        SurfaceMeshWebApi,
        VolumeMeshWebApi,
    )

    webapi_by_type = {
        "Geometry": GeometryWebApi,
        "SurfaceMesh": SurfaceMeshWebApi,
        "VolumeMesh": VolumeMeshWebApi,
        "Case": CaseWebApi,
        "Draft": DraftWebApi,
    }

    webapi_cls = webapi_by_type.get(resource_type)
    if webapi_cls is None:
        raise ResourceRefError(
            f"Opening {resource_type} resources in the browser is not supported."
        )

    return webapi_cls(resource_id).get_info()


def _get_workspace_id_for_root_folder(root_folder_id: str) -> str | None:
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.workspace_webapi import WorkspaceWebApi

    return WorkspaceWebApi.get_workspace_id_for_root_folder(root_folder_id)


def _get_root_folder_id(resource_id: str) -> str:
    if _is_root_folder_id(resource_id):
        return resource_id

    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    current_id = resource_id
    while True:
        info = FolderWebApi(current_id).get_info()
        parent_folders = info.get("parentFolders") or []
        for ancestor in parent_folders:
            ancestor_id = ancestor.get("id")
            if ancestor_id and _is_root_folder_id(ancestor_id):
                return ancestor_id

        parent_id = info.get("parentFolderId")
        if not parent_id:
            return current_id
        if _is_root_folder_id(parent_id):
            return parent_id
        current_id = parent_id


def _resolve_folder_workspace_id(resource_id: str) -> str:
    root_folder_id = _get_root_folder_id(resource_id)
    root_workspace_id = _get_workspace_id_for_root_folder(root_folder_id)
    if root_workspace_id:
        return root_workspace_id

    raise ResourceRefError(
        f"Could not infer a workspace for folder {resource_id}. "
        f"No workspace matched rootFolderId {root_folder_id}."
    )


def _get_folder_browser_path(resource_id: str, workspace_id: str | None) -> str:
    resolved_workspace_id = workspace_id or _resolve_folder_workspace_id(resource_id)
    query = urlencode(
        {
            "workspaceId": resolved_workspace_id,
            "folderId": resource_id,
            "activeTabIndex": 0,
        }
    )
    return f"workspaces?{query}"


def _get_workbench_path(project_id: str, resource_id: str, resource_type: str) -> str:
    query = urlencode({"id": resource_id, "type": resource_type})
    return f"workbench/{project_id}?{query}"


def get_resource_browser_payload(ref_id: str, *, workspace_id: str | None = None) -> dict:
    """Resolve a typed Flow360 ref to a browser-openable URL payload."""
    resource_ref = parse_resource_ref(ref_id)
    if resource_ref.resource_type == "Project":
        path = f"workbench/{resource_ref.id}"
    elif resource_ref.resource_type == "Folder":
        path = _get_folder_browser_path(resource_ref.id, workspace_id)
    else:
        info = _get_project_scoped_resource_info(resource_ref.resource_type, resource_ref.id)
        project_id = info.get("projectId")
        if not project_id:
            raise ResourceRefError(
                f"{resource_ref.resource_type} {resource_ref.id} does not expose a projectId."
            )
        path = _get_workbench_path(project_id, resource_ref.id, resource_ref.resource_type)
    url = Env.current.get_web_real_url(path)
    return {
        "id": resource_ref.id,
        "type": resource_ref.resource_type,
        "url": url,
    }


def open_browser_url(url: str) -> bool:
    """Best-effort browser open that never raises CLI-visible browser errors."""
    try:
        return bool(webbrowser.open(url))
    except webbrowser.Error:
        return False
