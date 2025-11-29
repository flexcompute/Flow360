"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
import copy
from typing import Dict, List, Optional, Union

from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import GeometryBodyGroup, Surface
from flow360.component.simulation.draft_context.mirror import (
    MirrorPlane,
    MirroredGeometryBodyGroup,
    MirroredSurface,
)
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360RuntimeError
from flow360.log import log


__all__ = [
    "DraftContext",
    "create_draft",
    "get_active_draft",
    "capture_into_draft",
]


_ACTIVE_DRAFT: ContextVar[DraftContext | None] = ContextVar("_ACTIVE_DRAFT", default=None)


def get_active_draft() -> DraftContext | None:
    """Return the current active draft context if any."""
    return _ACTIVE_DRAFT.get()


def create_draft(*, new_run_from: Union[Project]) -> DraftContext:
    """Factory helper used by end users (`with fl.create_draft() as draft`)."""
    # Get the entity info from the `new_run_ from` asset.
    root_asset = new_run_from.root_asset
    entity_info = root_asset.get_entity_info()
    return DraftContext()


# class _MirrorActionOnEntities(Flow360BaseModel):
#     """Action to mirror a GeometryBodyGroup entity."""

#     #TODO: Actually may not need Flow360BaseModel since it is too powerful for this simple case.
#     type_name: Literal["MirrorActionOnEntities"] = pd.Field("MirrorActionOnEntities", frozen=True)
#     geometry_body_group_ids: List[str] = pd.Field(description="List of GeometryBodyGroup IDs to mirror.")
#     mirror_plane_id: str = pd.Field(description="ID of the MirrorPlane to mirror the GeometryBodyGroup entities.")


 class _SingleTypeEntityRegistry(EntityRegistry):
    """
    Entity registry for a single type of entity.
    """
    def __init__(self) -> None:
        ...
    def __getitem__(self, key: str) -> EntityBase:
        """
        Supporting syntax like my_draft.body_groups["body_group_1"] and also my_draft.body_groups["body_group*"]
        Only glob patter is supported for now.
        """
        ...
        find_by_naming_pattern(pattern, use_glob_only=True)


class DraftContext(AbstractContextManager["DraftContext"]):
    """Context manager that tracks locally modified simulation entities."""

    def __init__(self, entity_info) -> None:
        """
        Data members:
        - _token: Token to track the active draft context.

        - _mirror_actions: Dictionary to track the mirror actions.
        The key is the GeometryBodyGroup ID and the value is MirrorPlane ID to mirror.

        - _mirror_planes: List to track the MirrorPlane entities.

        - _entity_registry: Registry of entities captured into the draft (optional).
        """
        self._token: Optional[Token] = None

        self._mirror_actions: Dict[str, str] = {}
        self._mirror_planes: List[MirrorPlane] = []

        self._entity_info = copy.deepcopy(entity_info)

        # Persistent entities (referencing objects in the _entity_info)
        self._body_groups:_SingleTypeEntityRegistry = ...
        self._surfaces:_SingleTypeEntityRegistry = ...
        self._edges:_SingleTypeEntityRegistry = ...
        self._volumes:_SingleTypeEntityRegistry = ...
        self._boxes:_SingleTypeEntityRegistry = ...
        self._cylinders:_SingleTypeEntityRegistry = ...

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

    # region -----------------------------Public properties Below-------------------------------------
    # Persistent entities
    @property
    def body_groups(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of body groups in the draft.


        Example
        -------
          >>> with fl.create_draft(new_run_from=geometry) as draft:
          >>>     draft.body_groups["body_group_1"]
          >>>     draft.body_groups["body_group*"]
        
        ====
        """
        ...

    @property
    def surfaces(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of surfaces in the draft.
        """
        ...

    @property
    def edges(self) -> _SingleTypeEntityRegistry:
        ...

    @property
    def volumes(self) -> _SingleTypeEntityRegistry:
        # If volume zone as root asset.
        ...
    
    # Non persistent entities
    @property
    def boxes(self) -> _SingleTypeEntityRegistry:
        ...
    @property
    def cylinders(self) -> _SingleTypeEntityRegistry:
        ...

    # endregion ------------------------------------------------------------------------------------

    # region -----------------------------Public Methods Below-------------------------------------
    def mirror(
        self, *, entities: List[EntityBase], mirror_plane: MirrorPlane
    ) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
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
        normalized_entities: list[EntityBase]
        if isinstance(entities, EntityBase):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360RuntimeError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        geometry_body_groups: list[GeometryBodyGroup] = []
        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360RuntimeError(
                    "Only GeometryBodyGroup entities are supported by `mirror()` currently. "
                    f"Received: {type(entity).__name__}."
                )
            geometry_body_groups.append(entity)

        # 2. [Validation] Ensure `mirror_plane` is a `MirrorPlane` entity.
        if not is_exact_instance(mirror_plane, MirrorPlane):
            raise Flow360RuntimeError(
                f"`mirror_plane` must be a MirrorPlane entity. Instead received: {type(mirror_plane).__name__}."
            )

        # 3. [Restriction] Each GeometryBodyGroup entity can only be mirrored once.
        #                  If a duplicate request is made, reset to the new one with a warning.
        for body_group in geometry_body_groups:
            body_group_id = body_group.private_attribute_id
            if body_group_id in self._mirror_actions:
                log.warning(
                    "GeometryBodyGroup `%s` was already mirrored; resetting to the latest mirror plane request.",
                    body_group.name,
                )

        # 3.2[Restriction] Face grouping should not clash with body grouping? (Waiting for PM's confirmation)
        #    (Intentionally left blank pending PM requirements.)

        # 4. Create/Update the self._mirror_actions
        #    and also capture the MirrorPlane into the `draft`.
        for body_group in geometry_body_groups:
            self._mirror_actions[body_group.private_attribute_id] = (
                mirror_plane.private_attribute_id
            )

        existing_plane_ids = {plane.private_attribute_id for plane in self._mirror_planes}
        if mirror_plane.private_attribute_id not in existing_plane_ids:
            self._mirror_planes.append(mirror_plane)

        # 5. Derive the generated mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
        #    and return to user as tokens of use.
        def _add_mirror_suffix(name: str) -> str:
            # suffix = mirror_plane.name or "mirror"
            suffix = "<mirror>"
            return f"{name}_{suffix}"

        mirrored_geometry_groups = [
            MirroredGeometryBodyGroup(
                name=_add_mirror_suffix(body_group.name),
                geometry_body_group_id=body_group.private_attribute_id,
                mirror_plane_id=mirror_plane.private_attribute_id,
            )
            for body_group in geometry_body_groups
        ]

        mirrored_surfaces: list[MirroredSurface] = []
        surface_candidates: list[Surface] = []
        registry = getattr(self, "_entity_registry", None)
        if registry is not None:
            try:
                surface_candidates = registry.get_bucket(by_type=Surface).entities
            except Exception:  # pylint: disable=broad-exception-caught
                surface_candidates = []

        def _extract_body_ids_from_surface(surface: Surface) -> set[str]:
            body_id_candidates: set[str] = set()
            for face_id in surface.private_attribute_sub_components or []:
                if "::" in face_id:
                    body_id_candidates.add(face_id.split("::", 1)[0])
                elif "_" in face_id:
                    body_id_candidates.add(face_id.split("_", 1)[0])
                else:
                    body_id_candidates.add(face_id)
            return body_id_candidates

        mirrored_surface_ids: set[str] = set()
        for candidate_surface in surface_candidates:
            surface_body_ids = _extract_body_ids_from_surface(candidate_surface)
            if not surface_body_ids:
                continue

            if len(surface_body_ids) > 1:
                log.warning(
                    "Surface `%s` spans multiple body groups (%s); mirrored copy may overlap.",
                    candidate_surface.name,
                    ", ".join(sorted(surface_body_ids)),
                )

            if candidate_surface.private_attribute_id in mirrored_surface_ids:
                continue

            mirrored_surface_ids.add(candidate_surface.private_attribute_id)
            mirrored_surfaces.append(
                MirroredSurface(
                    name=_add_mirror_suffix(candidate_surface.name),
                    surface_id=candidate_surface.private_attribute_id,
                    mirror_plane_id=mirror_plane.private_attribute_id,
                )
            )

        return mirrored_geometry_groups, mirrored_surfaces

    # endregion -------------------------------------------------------------------------------------
