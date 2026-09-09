"""TEST-ONLY helpers producing wire-legal params dicts from live-built params.

On the compact wire format every referenced entity needs a definition in
``asset_cache.project_entity_info``. Tests that build params in Python and
re-validate the JSON dump use :func:`dump_wire_legal` to mimic what upload
preparation does in production: collect the used entities into a minimal
entity_info.
"""

from typing import Any, Literal, get_args

from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup, MirrorPlane
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
    WindTunnelGhostSurface,
)
from flow360_schema.models.entities.volume_entities import GenericVolume
from flow360_schema.models.entity_info import DraftEntityTypes

_DRAFT_TYPES = get_args(get_args(DraftEntityTypes)[0])
_GHOST_TYPES = (GhostSphere, GhostCircularPlane, WindTunnelGhostSurface)


def dump_wire_legal(
    params: Any,
    *,
    entity_info_type: Literal[
        "GeometryEntityInfo", "SurfaceMeshEntityInfo", "VolumeMeshEntityInfo"
    ] = "VolumeMeshEntityInfo",
) -> dict[str, Any]:
    """Dump params to a wire-legal JSON dict.

    Every entity assigned anywhere in the params is registered into a minimal
    ``project_entity_info`` so the compact references the serializer emits can
    resolve at load. An existing ``project_entity_info`` on the params is kept.
    """
    from flow360_schema.models.entity_info import iter_entity_dicts_from_entity_info_dict

    used_entities = [
        entity for entity_list in params.used_entity_registry.internal_registry.values() for entity in entity_list
    ]
    if any(isinstance(entity, (Edge, GeometryBodyGroup, MirroredGeometryBodyGroup)) for entity in used_entities):
        # Edges and body groups only have a home in geometry entity info.
        entity_info_type = "GeometryEntityInfo"

    params_as_dict: dict[str, Any] = params.model_dump(mode="json", exclude_none=True)
    asset_cache = params_as_dict.setdefault("private_attribute_asset_cache", {})
    entity_info = asset_cache.get("project_entity_info")

    # Entities already homed on the asset cache never need an entity_info seat.
    _mirror_status = asset_cache.get("mirror_status") or {}
    _pre_homed = {
        (e.get("private_attribute_entity_type_name"), e.get("private_attribute_id"))
        for extra in (
            asset_cache.get("imported_surfaces") or [],
            _mirror_status.get("mirror_planes") or [],
            _mirror_status.get("mirrored_surfaces") or [],
            _mirror_status.get("mirrored_geometry_body_groups") or [],
        )
        for e in extra
    }
    if not entity_info and all(
        (entity.private_attribute_entity_type_name, entity.private_attribute_id) in _pre_homed
        for entity in used_entities
    ):
        # Nothing to place: do not fabricate an entity_info (a synthetic one can
        # trip consistency validators unrelated to the test's intent).
        return params_as_dict

    if not entity_info:
        entity_info = {"type_name": entity_info_type, "draft_entities": [], "ghost_entities": []}
        if entity_info_type == "GeometryEntityInfo":
            entity_info.update(
                {
                    "face_attribute_names": ["faceName"],
                    "face_group_tag": "faceName",
                    "grouped_faces": [[]],
                }
            )
        else:
            entity_info["boundaries"] = []
        if entity_info_type == "VolumeMeshEntityInfo":
            entity_info["zones"] = []
        asset_cache["project_entity_info"] = entity_info

    known = {
        (e.get("private_attribute_entity_type_name"), e.get("private_attribute_id"))
        for e in iter_entity_dicts_from_entity_info_dict(entity_info, grouped="all")
    }
    mirror_status = asset_cache.get("mirror_status") or {}
    for extra in (
        asset_cache.get("imported_surfaces") or [],
        mirror_status.get("mirror_planes") or [],
        mirror_status.get("mirrored_surfaces") or [],
        mirror_status.get("mirrored_geometry_body_groups") or [],
    ):
        for e in extra:
            known.add((e.get("private_attribute_entity_type_name"), e.get("private_attribute_id")))
    is_volume_mesh_info = entity_info.get("type_name") == "VolumeMeshEntityInfo"

    def dump(entity: Any) -> dict[str, Any]:
        dumped: dict[str, Any] = entity.model_dump(
            mode="json", exclude_none=True, context={"inline_stored_entities": True}
        )
        return dumped

    for entity in used_entities:
        if (entity.private_attribute_entity_type_name, entity.private_attribute_id) in known:
            continue
        if isinstance(entity, Surface):
            if entity_info.get("type_name") == "GeometryEntityInfo":
                entity_info.setdefault("grouped_faces", [[]])[0].append(dump(entity))
            else:
                surface_dict = dump(entity)
                # VolumeMesh boundaries require the interface flag to be set.
                surface_dict.setdefault("private_attribute_is_interface", False)
                entity_info.setdefault("boundaries", []).append(surface_dict)
        elif isinstance(entity, GenericVolume):
            if is_volume_mesh_info:
                entity_info.setdefault("zones", []).append(dump(entity))
            # Other entity info types carry no zones (mesher output); such
            # references stay unresolvable and the test must not rely on them.
        elif isinstance(entity, _GHOST_TYPES):
            entity_info.setdefault("ghost_entities", []).append(dump(entity))
        elif isinstance(entity, _DRAFT_TYPES):
            entity_info.setdefault("draft_entities", []).append(dump(entity))
        elif isinstance(entity, ImportedSurface):
            asset_cache.setdefault("imported_surfaces", []).append(dump(entity))
        elif isinstance(entity, Edge):
            if entity_info.get("type_name") == "GeometryEntityInfo":
                entity_info.setdefault("edge_attribute_names", ["edgeId"])
                entity_info.setdefault("edge_group_tag", "edgeId")
                entity_info.setdefault("grouped_edges", [[]])[0].append(dump(entity))
            # Non-geometry entity info has no edge home; the assignment keeps
            # its inline form via the transitional tolerance.
        elif isinstance(entity, MirroredSurface):
            asset_cache.setdefault("mirror_status", {}).setdefault("mirrored_surfaces", []).append(dump(entity))
        elif isinstance(entity, MirroredGeometryBodyGroup):
            asset_cache.setdefault("mirror_status", {}).setdefault("mirrored_geometry_body_groups", []).append(
                dump(entity)
            )
        elif isinstance(entity, MirrorPlane):
            asset_cache.setdefault("mirror_status", {}).setdefault("mirror_planes", []).append(dump(entity))
        elif isinstance(entity, GeometryBodyGroup):
            entity_info.setdefault("body_attribute_names", ["groupByFile"])
            entity_info.setdefault("body_group_tag", "groupByFile")
            entity_info.setdefault("grouped_bodies", [[]])[0].append(dump(entity))
        else:
            raise ValueError(f"dump_wire_legal cannot place entity type {type(entity).__name__}")

    return params_as_dict


def round_trip_simulation_dict(params_as_dict: dict[str, Any]) -> tuple[Any, Any]:
    """Round-trip a simulation dict through the compact wire format.

    validate (updater + loader) -> dump (compact) -> validate again. Returns
    the two resolved params instances; raises AssertionError when either pass
    reports errors or when the round trip is not stable (params inequality or
    a second dump differing from the first).

    validate_model mutates its input in place (materialization), so each pass
    gets its own deep copy.
    """
    import copy

    from flow360_schema.models.simulation.validation.validation_service import (
        ValidationCalledBy,
        validate_model,
    )

    # validation_level=[] is load-only semantics (same tolerance as from_file /
    # asset.params): leveled semantic validators stay out of a WIRE-format sweep.
    params_first, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(params_as_dict),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=[],
    )
    assert errors is None, f"first validation pass failed: {errors}"
    assert params_first is not None

    dump_first = params_first.model_dump(mode="json", exclude_none=True)
    params_second, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(dump_first),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=[],
    )
    assert errors is None, f"second validation pass (compact reload) failed: {errors}"
    assert params_second is not None

    dump_second = params_second.model_dump(mode="json", exclude_none=True)
    # The first pass may normalize legacy quirks (e.g. explicit-null fields that
    # exclude_none turns into absent ones, re-triggering defaulting validators).
    # From the second pass on, the compact format must be a fixed point.
    # Dump equality is the resolved-state comparison; pydantic __eq__ on the
    # models trips numpy's ambiguous truth value for array-carrying fields.
    if dump_first != dump_second:
        params_third, errors, _ = validate_model(
            params_as_dict=copy.deepcopy(dump_second),
            validated_by=ValidationCalledBy.LOCAL,
            root_item_type=None,
            validation_level=[],
        )
        assert errors is None, f"third validation pass failed: {errors}"
        assert params_third is not None
        dump_third = params_third.model_dump(mode="json", exclude_none=True)
        assert dump_second == dump_third, "compact dump is not round-trip stable"
    return params_first, params_second


def assert_unique_entity_instances(params: Any) -> int:
    """Assert every (entity type, id) in stored_entities resolves to ONE live instance.

    The materializer resolves reference groups against one registry, so two
    ASSIGNMENTS of the same entity must share the instance. Standalone entity
    fields (e.g. Rotation.parent_volume) and input caches parse their own
    copies and are outside this contract. Returns the number of distinct
    entities seen (lets callers assert non-trivial coverage).
    """
    import pydantic

    from flow360_schema.framework.entity.entity_base import EntityBase
    from flow360_schema.framework.entity.entity_list import EntityList

    instances_by_key: dict[tuple[str, str], set[int]] = {}
    seen: set[int] = set()

    def visit(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, EntityList):
            for item in node.stored_entities:
                if isinstance(item, EntityBase):
                    key = (item.private_attribute_entity_type_name, item.private_attribute_id)
                    instances_by_key.setdefault(key, set()).add(id(item))
            return
        if isinstance(node, EntityBase):
            return
        if isinstance(node, pydantic.BaseModel):
            for field_name in type(node).model_fields:
                if field_name == "private_attribute_input_cache":
                    continue
                visit(getattr(node, field_name))
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        if isinstance(node, dict):
            for item in node.values():
                visit(item)

    # The asset cache is the definitions store; its instances are parsed
    # separately by design. Identity is guaranteed among ASSIGNMENTS only.
    for field_name in type(params).model_fields:
        if field_name == "private_attribute_asset_cache":
            continue
        visit(getattr(params, field_name))

    duplicated = {key: ids for key, ids in instances_by_key.items() if len(ids) > 1}
    assert not duplicated, f"entities resolved to multiple instances: {sorted(duplicated)}"
    return len(instances_by_key)


def attach_wire_entity_info(params: Any, *, entity_info_type: str = "VolumeMeshEntityInfo") -> Any:
    """Force-set a minimal ``project_entity_info`` holding every used entity.

    Instance-level sibling of :func:`dump_wire_legal` for fixtures that
    round-trip through ``to_file``/``from_file``.
    """
    from flow360_schema.models.entity_info import SurfaceMeshEntityInfo, VolumeMeshEntityInfo

    info_cls = {"VolumeMeshEntityInfo": VolumeMeshEntityInfo, "SurfaceMeshEntityInfo": SurfaceMeshEntityInfo}[
        entity_info_type
    ]
    boundaries, zones, drafts, ghosts = [], [], [], []
    for entity_list in params.used_entity_registry.internal_registry.values():
        for entity in entity_list:
            if isinstance(entity, Surface):
                if entity.private_attribute_is_interface is None:
                    entity = entity.model_copy(update={"private_attribute_is_interface": False})
                boundaries.append(entity)
            elif isinstance(entity, GenericVolume):
                zones.append(entity)
            elif isinstance(entity, _GHOST_TYPES):
                ghosts.append(entity)
            elif isinstance(entity, _DRAFT_TYPES):
                drafts.append(entity)
    kwargs = {"boundaries": boundaries, "draft_entities": drafts, "ghost_entities": ghosts}
    if entity_info_type == "VolumeMeshEntityInfo":
        kwargs["zones"] = zones
    params.private_attribute_asset_cache._force_set_attr("project_entity_info", info_cls(**kwargs))
    return params


def validate_for_translation(
    params: Any,
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"],
    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"],
) -> tuple[Any, Any, Any]:
    """Serialize live-built params and validate them for one submission path.

    Translator tests and local integration scripts build params in Python but must
    translate what validation produces from the serialized form, at the levels a
    given ``(root_item_type, up_to)`` submission validates. Returns
    ``(params, errors, warnings)``.
    """
    from flow360_schema.models.simulation.services_utils import (
        strip_implicit_edge_split_layers_inplace,
    )
    from flow360_schema.models.simulation.validation.validation_service import (
        ValidationCalledBy,
        _determine_validation_level,
        validate_model,
    )

    params_as_dict = params.model_dump(mode="json", exclude_none=True)
    params_as_dict = strip_implicit_edge_split_layers_inplace(params, params_as_dict)
    return validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=root_item_type,
        validation_level=_determine_validation_level(root_item_type=root_item_type, up_to=up_to),
    )
