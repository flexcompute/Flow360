"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Dict, List, Optional

from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.exceptions import Flow360RuntimeError
from flow360.component.simulation.draft_context.mirror import MirrorPlane


__all__ = [
    "DraftContext",
    "create_draft",
    "get_active_draft",
]


_ACTIVE_DRAFT: ContextVar[DraftContext | None] = ContextVar("_ACTIVE_DRAFT", default=None)


def get_active_draft() -> DraftContext | None:
    """Return the current active draft context if any."""
    return _ACTIVE_DRAFT.get()


def create_draft() -> DraftContext:
    """Factory helper used by end users (`with fl.create_draft() as draft`)."""
    return DraftContext()


# class _MirrorActionOnEntities(Flow360BaseModel):
#     """Action to mirror a GeometryBodyGroup entity."""

#     #TODO: Actually may not need Flow360BaseModel since it is too powerful for this simple case.
#     type_name: Literal["MirrorActionOnEntities"] = pd.Field("MirrorActionOnEntities", frozen=True)
#     geometry_body_group_ids: List[str] = pd.Field(description="List of GeometryBodyGroup IDs to mirror.")
#     mirror_plane_id: str = pd.Field(description="ID of the MirrorPlane to mirror the GeometryBodyGroup entities.")


class DraftContext(AbstractContextManager["DraftContext"]):
    """Context manager that tracks locally modified simulation entities."""

    def __init__(self) -> None:
        """
        Data members:
        - _token: Token to track the active draft context.

        - _mirror_actions: Dictionary to track the mirror actions.
        The key is the GeometryBodyGroup ID and the value is MirrorPlane ID to mirror.

        - _mirror_planes: List to track the MirrorPlane entities.
        """
        self._token: Optional[Token] = None

        self._mirror_actions: Optional[Dict[str, str]] = None
        self._mirror_planes: Optional[List[MirrorPlane]] = None

    def __enter__(self) -> DraftContext:
        if get_active_draft() is not None:
            raise Flow360RuntimeError("Nested draft contexts are not allowed.")
        self._token = _ACTIVE_DRAFT.set(self)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._token is None:
            raise Flow360RuntimeError(
                "[Internal] DraftContext exit called without a matching enter."
            )
        _ACTIVE_DRAFT.reset(self._token)
        self._token = None
        return False

    # region -----------------------------Private implementations Below-----------------------------

    def _update_mirror_status_into_asset_cache(self) -> None:
        """
        Before submission, serialize the mirror status into the asset cache.
        """
        # TODO: Get front end's requirement on the schema.
        ...

    def _read_mirror_status_from_asset_cache(self, param_as_dict: dict) -> None:
        """
        Deserialize the mirror status from the asset cache.
        """
        # TODO: Get front end's requirement on the schema.
        ...

    # endregion ------------------------------------------------------------------------------------

    # region -----------------------------Public Methods Below-------------------------------------

    def mirror(self, entities: List[EntityBase], mirror_plane: MirrorPlane):
        """
        Create mirrored GeometryBodyGroup (and its associated surfaces) for the given `MirrorPlane`.
        New entities will have "_<mirror>" in the name as suffix.

        Example
        -------
          >>> with fl.create_draft() as draft:
          >>>     mirror_plane = fl.MirrorPlane(center=(0, 0, 0)*fl.u.m, normal=(1, 0, 0))
          >>>     draft.mirror(entities=geometry["body1"], mirror_plane=mirror_plane)

        ====
        """

        # pylint: disable=fixme
        # TODO: Support EntitySelector for specifying the GeometryBodyGroup in the future?

        # 1. [Validation] Ensure `entities` are GeometryBodyGroup entities.

        # 2. [Validation] Ensure `mirror_plane` is a `MirrorPlane` entity.

        # 3. [Restriction] Each GeometryBodyGroup entity can only be mirrored once.
        #                  Raise warning if the entity has already been mirrored.

        # 3.2[Restriction] Face grouping should not clash with body grouping? (Waiting for PM's confirmation)

        # 4. Create/Update the self._mirror_actions (AI: Can you come up with a better name?)
        #    and also capture the MirrorPlane into the `draft`.

        # 5. Derive the generated mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
        #    and return to user as tokens of use.

    # endregion -------------------------------------------------------------------------------------
