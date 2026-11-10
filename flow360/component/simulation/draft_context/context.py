"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

import warnings
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, List, Optional, Union, get_args

from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_expansion_config import (
    DEFAULT_TARGET_CLASS_EXPANSION_MAP,
)
from flow360_schema.framework.entity.entity_registry import (
    EntityRegistry,
    EntityRegistryView,
)
from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.framework.entity.entity_utils import get_entity_key
from flow360_schema.models.asset_cache import CoordinateSystemStatus, MirrorStatus
from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup
from flow360_schema.models.entities.surface_entities import (
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
)
from flow360_schema.models.entities.volume_entities import GenericVolume
from flow360_schema.models.entity_info import (
    DraftEntityTypes,
    EntityInfoModel,
    GeometryEntityInfo,
)
from flow360_schema.models.simulation.framework.updater_utils import (
    deprecation_reminder,
)

from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemManager,
)
from flow360.component.simulation.draft_context.mirror import MirrorManager
from flow360.component.simulation.framework.entity_expansion_utils import (
    EntitySelection,
    infer_target_class,
    normalize_selection,
)
from flow360.exceptions import Flow360RuntimeError, Flow360ValueError
from flow360.log import log

if TYPE_CHECKING:
    from flow360.component.geometry import Geometry

__all__ = [
    "DraftContext",
    "get_active_draft",
]


_ACTIVE_DRAFT: ContextVar[Optional["DraftContext"]] = ContextVar("_ACTIVE_DRAFT", default=None)

_DRAFT_ENTITY_TYPE_TUPLE: tuple[type[EntityBase], ...] = tuple(
    get_args(get_args(DraftEntityTypes)[0])
)


def get_active_draft() -> Optional["DraftContext"]:
    """Return the current active draft context if any."""
    return _ACTIVE_DRAFT.get()


class DraftContext(  # pylint: disable=too-many-instance-attributes
    AbstractContextManager["DraftContext"]
):
    """
    Context manager that tracks locally modified simulation entities/status.
    This should (eventually, not right now) be replacement of accessing entities directly from assets.
    """

    __slots__ = (
        # Persistent entities data storage.
        "_entity_info",
        # Interface accessing ALL types of entities.
        "_entity_registry",
        "_imported_surfaces",
        "_imported_geometries",
        # Lightweight mirror relationships storage (compared to entity storages)
        "_mirror_manager",
        # Internal mirror related entities data storage.
        "_mirror_status",
        # Lightweight coordinate system relationships storage (compared to entity storages)
        "_coordinate_system_manager",
        # Geometry root resource from which this draft was created.
        "_geometry_root",
        # Beta notices already shown for this draft, so loops do not repeat them.
        "_beta_notices_shown",
        # Rule-based face grouping carried as draft state (persisted at submission)
        "_face_grouping_rules",
        # Lazy provider of the source geometry's topology tree (geometry-root
        # drafts only) — what preview_face_grouping resolves against. A provider,
        # not the tree: fetching is a cloud call that most drafts never need.
        "_topology_tree_provider",
        "_token",
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        entity_info: EntityInfoModel,
        entity_registry: Optional[EntityRegistry] = None,
        imported_geometries: Optional[List] = None,
        imported_surfaces: Optional[List[ImportedSurface]] = None,
        mirror_status: Optional[MirrorStatus] = None,
        coordinate_system_status: Optional[CoordinateSystemStatus] = None,
        geometry_root: Optional[Geometry] = None,
        face_grouping_rules=None,
        topology_tree_provider=None,
    ) -> None:
        """
        Data members:
        - _token: Token to track the active draft context.

        - _mirror_manager: Manager for mirror planes and mirrored entities.

        - _entity_registry: Registry of entities of self._entity_info.
                            This provides interface for user to access the entities in the draft.
                            A prebuilt registry may be passed in, in which case it MUST have been
                            built from the exact `entity_info` instance handed over (shared
                            instances are the draft's identity guarantee).

        """

        if entity_info is None:
            raise Flow360RuntimeError(
                "[Internal] DraftContext requires `entity_info` to initialize."
            )
        self._token: Optional[Token] = None
        self._geometry_root = geometry_root
        self._beta_notices_shown: set = set()
        # Rules are draft state, not a cloud call: they ride the normal submission
        # flow (run/draft_only) inside simulation.json's asset cache, never a
        # per-feature save method. The resolved groups live in _entity_info.
        self._face_grouping_rules = face_grouping_rules
        self._topology_tree_provider = topology_tree_provider

        # The caller transfers ownership of entity_info (and the registry when prebuilt):
        # it must be an instance no other owner mutates, e.g. a fresh deserialization.
        self._entity_info = entity_info
        self._entity_registry: EntityRegistry = (
            entity_registry
            if entity_registry is not None
            else EntityRegistry.from_entity_info(entity_info)
        )

        self._imported_surfaces: List = imported_surfaces or []
        known_frozen_hashes = set()
        for imported_surface in self._imported_surfaces:
            known_frozen_hashes = self._entity_registry.fast_register(
                imported_surface, known_frozen_hashes
            )
        self._imported_geometries: List = imported_geometries if imported_geometries else []
        # Pre-compute face_group_to_body_group map for mirror operations.
        # This is only available for GeometryEntityInfo.
        face_group_to_body_group = None

        if isinstance(self._entity_info, GeometryEntityInfo):
            try:
                face_group_to_body_group = self._entity_info.get_face_group_to_body_group_id_map()
            except Flow360ValueError as exc:
                # Face grouping spans across body groups.
                log.warning(
                    "Failed to derive surface-to-body-group mapping for mirroring: %s. "
                    "Mirroring will be disabled.",
                    exc,
                )
        self._mirror_manager = MirrorManager._from_status(
            status=mirror_status,
            face_group_to_body_group=face_group_to_body_group,
            entity_registry=self._entity_registry,
        )

        self._coordinate_system_manager = CoordinateSystemManager._from_status(
            status=coordinate_system_status,
            entity_registry=self._entity_registry,
        )

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

    def _warn_beta_once(self, feature: str) -> None:
        """Emit the beta notice for `feature` at most once per draft.

        These helpers get called in loops -- one selector per boundary condition is
        normal -- so warning per call buries the rest of the output.
        """
        if feature in self._beta_notices_shown:
            return
        self._beta_notices_shown.add(feature)
        log.warning(
            f"!!! `{feature}` is a beta feature and may be removed or changed "
            "in future releases. !!!"
        )

    # endregion ------------------------------------------------------------------------------------

    # region -----------------------------Public properties Below-------------------------------------

    @property
    def face_grouping_rules(self):
        """The rule-based face-grouping payload this draft carries. Raises when the
        draft uses a plain grouping tag. Persisted automatically at submission (the
        resolved assignment is internal draft state, not API surface).

        Read-only: rules are chosen at create_draft(face_grouping=...) and the
        resolved groups are materialized from them right there, so re-pointing this
        attribute afterwards could not move the draft's surfaces with it."""
        if self._face_grouping_rules is not None:
            return self._face_grouping_rules
        active_tag = getattr(self._entity_info, "face_group_tag", None)
        if active_tag == "rule-based-grouping":
            raise Flow360RuntimeError(
                "This draft's grouping uses the 'rule-based-grouping' tag but carries no "
                "rules payload — the rules were authored elsewhere (e.g. the Web UI). "
                "Re-author them via create_draft(face_grouping=FaceGroupingRules(...)) "
                "to work with them here."
            )
        raise Flow360RuntimeError(
            "This draft does not carry rule-based face-grouping rules "
            f"(grouping tag: '{active_tag}'). Author them via "
            "create_draft(face_grouping=FaceGroupingRules(...))."
        )

    def preview_face_grouping(self, face_grouping=None) -> dict:
        """Resolve face-grouping rules against this draft's topology tree,
        READ-ONLY — nothing is materialized or persisted. With no argument,
        previews the draft's own rules; pass a FaceGroupingRules payload to try
        an alternative before re-authoring the draft. Returns
        ``{group_name: [face_id, ...]}`` — the ``draft.surfaces`` names the same
        rules materialize (first-match-wins partition; overlaps warn loudly; leftover
        faces without an ``is_remaining`` group, a rule group left with no faces, or a
        group spanning several model files is refused, exactly as at create_draft)."""
        # pylint: disable=import-outside-toplevel
        from flow360_schema.geometry.selection_rules import resolve_group_assignments

        from flow360.component.geometry_face_grouping import faces_by_group

        if self._topology_tree_provider is None:
            raise Flow360RuntimeError(
                "preview_face_grouping needs a topology tree — only drafts created "
                "from a geometry root carry one."
            )
        if face_grouping is None:
            face_grouping = self.face_grouping_rules
        tree = self._topology_tree_provider()
        return faces_by_group(resolve_group_assignments(face_grouping, tree))

    # Persistent entities
    @property
    def body_groups(self) -> EntityRegistryView:
        """
        Return the list of body groups in the draft.


        Example
        -------
        >>> with fl.create_draft(new_run_from=geometry) as draft:
        ...     draft.body_groups["body_group_1"]
        ...     draft.body_groups["body_group*"]
        """
        return self._entity_registry.view(GeometryBodyGroup)

    @property
    def surfaces(self) -> EntityRegistryView:
        """
        Return the list of surfaces in the draft.
        """
        return self._entity_registry.view(Surface)

    @property
    def mirrored_body_groups(self) -> EntityRegistryView:
        """
        Return the list of mirrored body groups in the draft.

        Notes
        -----
        Mirrored entities are draft-only entities derived from mirror actions and stored in the draft registry.
        """
        return self._entity_registry.view(MirroredGeometryBodyGroup)

    @property
    def mirrored_surfaces(self) -> EntityRegistryView:
        """
        Return the list of mirrored surfaces in the draft.

        Notes
        -----
        Mirrored entities are draft-only entities derived from mirror actions and stored in the draft registry.
        """
        return self._entity_registry.view(MirroredSurface)

    @property
    def edges(self) -> EntityRegistryView:
        """
        Return the list of edges in the draft.
        """
        return self._entity_registry.view(Edge)

    @property
    def volumes(self) -> EntityRegistryView:
        """
        Return the list of volumes (volume zones) in the draft.
        """
        return self._entity_registry.view(GenericVolume)

    # Non-persistent entities
    @property
    def boxes(self) -> EntityRegistryView:
        """
        Return the list of boxes in the draft.
        """
        # pylint: disable=import-outside-toplevel
        from flow360_schema.models.entities.volume_entities import Box

        return self._entity_registry.view(Box)

    @property
    def cylinders(self) -> EntityRegistryView:
        """
        Return the list of cylinders in the draft.
        """
        # pylint: disable=import-outside-toplevel
        from flow360_schema.models.entities.volume_entities import Cylinder

        return self._entity_registry.view(Cylinder)

    @property
    def imported_geometries(self) -> List:
        """
        Return the list of imported geometries in the draft.
        """
        return self._imported_geometries

    @property
    def imported_surfaces(self) -> EntityRegistryView:
        """
        Return the list of imported surfaces in the draft.
        """
        return self._entity_registry.view(ImportedSurface)

    @property
    def coordinate_systems(self) -> CoordinateSystemManager:
        """
        Coordinate system manager for this draft.

        This is the primary user entry point to create/remove coordinate systems, define
        parent relationships, and assign coordinate systems to draft entities.

        See Also
        --------
        CoordinateSystemManager
        """
        return self._coordinate_system_manager

    @property
    def mirror(self) -> MirrorManager:
        """
        Mirror manager for this draft.

        This is the primary user entry point to define mirror planes and create/remove
        mirrored draft-only entities derived from geometry body groups.

        See Also
        --------
        MirrorManager
        """
        return self._mirror_manager

    def preview_selector(self, selector: EntitySelector, *, return_names: bool = True):
        """
        Preview which entities a selector would match in this draft context.

        Parameters
        ----------
        selector : EntitySelector
            The selector to preview (SurfaceSelector, EdgeSelector, VolumeSelector, or BodyGroupSelector).
        return_names : bool, default True
            When True, returns entity names. When False, returns entity instances.

        Returns
        -------
        list[str] or list[EntityBase]
            Matched entity names or instances depending on ``return_names``.

        Example
        -------
        >>> import flow360 as fl
        >>> geometry = fl.Geometry.from_cloud(id="...")
        >>> with fl.create_draft(new_run_from=geometry) as draft:
        ...     selector = fl.SurfaceSelector(name="wing_surfaces").match("wing*")
        ...     matched = draft.preview_selector(selector)
        ...     print(matched)  # ['wing_upper', 'wing_lower', ...]

        See Also
        --------
        preview_unselected : the complement -- entities no selection covers.
        """
        self._warn_beta_once("preview_selector")

        if not isinstance(selector, EntitySelector):
            raise Flow360ValueError(
                f"Expected EntitySelector, got {type(selector).__name__}. "
                "Use fl.SurfaceSelector, fl.EdgeSelector, fl.VolumeSelector, or fl.BodyGroupSelector."
            )

        matched_entities = normalize_selection(
            self._entity_registry, selector, operation="preview_selector()"
        )
        if return_names:
            return [entity.name for entity in matched_entities]
        return matched_entities

    def preview_unselected(self, selection: EntitySelection, *, return_names: bool = True):
        """
        Report which entities in this draft a selection does **not** cover.

        Catches silent under-selection: a glob matching nothing, or a name whose
        capitalization differs from the geometry, leaves surfaces out of every assignment
        without raising.

        The comparison pool is every entity the selection's kind of selector can reach,
        read from the configuration selector expansion itself uses. For surfaces that is
        ``Surface`` and ``MirroredSurface``; imported surfaces and ghost boundaries are
        not selectable and never reported. The pool spans the whole draft.

        Parameters
        ----------
        selection : EntitySelection
            What counts as selected. Must name exactly one entity kind and be non-empty,
            since the kind is inferred from it.
        return_names : bool, default True
            When True, returns entity names. When False, returns entity instances.

        Returns
        -------
        list[str] or list[EntityBase]
            Unselected entity names or instances depending on ``return_names``.

        Example
        -------
        >>> import flow360 as fl
        >>> geometry = fl.Geometry.from_cloud(id="...")
        >>> with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
        ...     wheels = fl.SurfaceSelector(name="wheels").match("*rim*")
        ...     body = fl.SurfaceSelector(name="body").match("*body*")
        ...     missed = draft.preview_unselected([wheels, body])
        ...     if missed:
        ...         raise ValueError(f"No selector covers: {missed}")

        See Also
        --------
        preview_selector : the complement -- what a single selector matches.
        """
        self._warn_beta_once("preview_unselected")
        operation = "preview_unselected()"

        target_class = infer_target_class(selection, operation=operation)
        selected = normalize_selection(self._entity_registry, selection, operation=operation)
        selected_keys = {get_entity_key(entity) for entity in selected}

        pool = self._entity_registry.find_by_type_name(
            DEFAULT_TARGET_CLASS_EXPANSION_MAP[target_class]
        )
        unselected = [entity for entity in pool if get_entity_key(entity) not in selected_keys]

        if return_names:
            return [entity.name for entity in unselected]
        return unselected

    @deprecation_reminder(
        "`draft.compute_obb()` is deprecated and will be replaced by a new API "
        "in a future release.",
    )
    def compute_obb(
        self,
        entities: Union[Surface, List[Surface], EntityRegistryView, EntitySelector],
    ):
        """Compute an oriented bounding box for selected surfaces.

        Deprecated: this method will be replaced by a new API in a future release.
        """
        warnings.warn(
            "`draft.compute_obb()` is deprecated and will be replaced by a new API "
            "in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.measurement import oriented_bounding_box

        return oriented_bounding_box(self, surfaces=entities)

    # endregion ------------------------------------------------------------------------------------
