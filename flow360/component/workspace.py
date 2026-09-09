"""Workspace component for targeting the shared company workspace."""

from __future__ import annotations

from typing import List, Literal, Optional

import pydantic as pd

from flow360.component.simulation.web.workspace_webapi import WorkspaceWebApi
from flow360.exceptions import Flow360ValueError


class Workspace(pd.BaseModel):
    """
    A Flow360 workspace. Every account has a PRIVATE workspace; accounts that belong
    to a company tenant also have a SHARED company workspace visible to all tenant members.

    Select one with :meth:`get_shared` or :meth:`get_private`, then pass it to the
    ``folder_or_workspace`` argument of the Project API or ``Folder.create()`` to
    create resources in it.

    Note
    ----
    Workspace selection is independent from account switching
    (:class:`~flow360.accounts_utils.AccountsUtils`). ``Accounts.choose_shared_account()``
    changes which user you act as; the workspace selects where resources are created
    for the currently effective user. When both are combined, select the workspace
    *after* switching accounts.
    """

    id: str
    name: str
    type: Literal["SHARED", "PRIVATE"]
    status: Optional[str] = None
    root_folder_id: str = pd.Field(alias="rootFolderId")

    model_config = pd.ConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def _get_all(cls) -> List[Workspace]:
        """List all workspaces available to the current account."""
        return [cls.model_validate(record) for record in WorkspaceWebApi.list_records()]

    @classmethod
    def _of_type(cls, workspace_type: str) -> Workspace:
        """Return the single workspace of the given type, listing the alternatives if there is none."""
        workspaces = cls._get_all()
        matches = [workspace for workspace in workspaces if workspace.type == workspace_type]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(f"{workspace.name!r} ({workspace.type})" for workspace in workspaces)
        raise Flow360ValueError(
            f"Expected exactly one {workspace_type} workspace, found {len(matches)}. "
            f"Available workspaces: {available}."
        )

    @classmethod
    def get_shared(cls) -> Workspace:
        """Get the shared company workspace of the current account."""
        return cls._of_type("SHARED")

    @classmethod
    def get_private(cls) -> Workspace:
        """Get the private workspace of the current account."""
        return cls._of_type("PRIVATE")
