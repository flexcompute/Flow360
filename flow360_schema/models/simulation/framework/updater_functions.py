from flow360_schema.framework.entity.entity_utils import generate_uuid

"""Implementation of the updater functions. The updated.py should just import functions from here."""


def fix_ghost_sphere_schema(*, params_as_dict: dict):
    """
    The previous ghost farfield has wrong schema (bug) and therefore needs data alternation.
    """

    def i_am_outdated_ghost_sphere(*, data: dict):
        """Identify if the current dict is a outdated ghost sphere."""
        if "type_name" in data and data["type_name"] == "GhostSphere":
            return True
        return False

    def recursive_fix_ghost_surface(*, data):
        if isinstance(data, dict):
            if i_am_outdated_ghost_sphere(data=data):
                data.pop("type_name")
                data["private_attribute_entity_type_name"] = "GhostSphere"

            for _, val in data.items():
                recursive_fix_ghost_surface(
                    data=val,
                )

        elif isinstance(data, list):
            for _, item in enumerate(data):
                recursive_fix_ghost_surface(data=item)

    recursive_fix_ghost_surface(data=params_as_dict)


def is_entity_dict(data: dict):
    """Check if current dict is an Entity item"""
    return data.get("name") and (data.get("private_attribute_entity_type_name") is not None)


def remove_entity_bucket_field(*, params_as_dict: dict):
    """Recursively remove legacy private_attribute_registry_bucket_name from all entity dicts."""

    def _recursive_remove(data):
        if isinstance(data, dict):
            if is_entity_dict(data=data):
                data.pop("private_attribute_registry_bucket_name", None)
            for value in data.values():
                _recursive_remove(value)
        elif isinstance(data, list):
            for element in data:
                _recursive_remove(element)

    _recursive_remove(params_as_dict)
    return params_as_dict


def populate_entity_id_with_name(*, params_as_dict: dict):
    """
    Recursively populates the entity item's private_attribute_id with its name if
    the private_attribute_id is none.
    """

    def recursive_populate_entity_id_with_name(*, data):
        if isinstance(data, dict):
            if is_entity_dict(data=data):
                if "private_attribute_id" not in data or data["private_attribute_id"] is None:
                    data["private_attribute_id"] = data["name"]

            for value in data.values():
                recursive_populate_entity_id_with_name(data=value)

        elif isinstance(data, list):
            for element in data:
                recursive_populate_entity_id_with_name(data=element)

    recursive_populate_entity_id_with_name(data=params_as_dict)


def update_symmetry_ghost_entity_name_to_symmetric(*, params_as_dict: dict):
    """
    Recursively update ghost entity name from symmetric-* to symmetry-*
    """

    def recursive_update_symmetry_ghost_entity_name_to_symmetric(*, data):
        if isinstance(data, dict):
            if (
                is_entity_dict(data=data)
                and data["private_attribute_entity_type_name"] == "GhostCircularPlane"
                and data["name"].startswith("symmetry")
            ):
                data["name"] = data["name"].replace("symmetry", "symmetric")

            for value in data.values():
                recursive_update_symmetry_ghost_entity_name_to_symmetric(data=value)

        elif isinstance(data, list):
            for element in data:
                recursive_update_symmetry_ghost_entity_name_to_symmetric(data=element)

    recursive_update_symmetry_ghost_entity_name_to_symmetric(data=params_as_dict)


# Types whose id was historically Optional and whose production convention is
# id == name (pipelines and ghost builders write name-based ids). Draft and
# mirror types have carried uuid default factories since their introduction, so
# an id-less one in stored JSON is an anomaly the updater must not paper over —
# it is left untouched and fails loudly at validation.
_LEGACY_OPTIONAL_ID_TYPE_NAMES = frozenset(
    {
        "Surface",
        "Edge",
        "GeometryBodyGroup",
        "GenericVolume",
        "GhostSurface",
        "GhostSphere",
        "GhostCircularPlane",
        "WindTunnelGhostSurface",
    }
)


def _backfill_entity_ids(params_as_dict):
    """Backfill the production id convention for legacy entity dicts.

    ``id := name`` for the historically-optional persistent and ghost types;
    ``{name}_defaultBody`` for ``ImportedSurface`` (whose handles are
    re-created from dependency metadata each session). Covers
    ``project_entity_info`` persistent lists (grouped faces/edges/bodies,
    boundaries, zones), ``ghost_entities``, ``draft_entities``, and inline
    ``stored_entities`` blobs anywhere in the params (including nested ones
    like ``CustomVolume.bounding_entities``).
    """
    entity_list_keys = ("stored_entities", "draft_entities", "ghost_entities", "boundaries", "zones")
    grouped_list_keys = ("grouped_faces", "grouped_edges", "grouped_bodies")

    def backfill(entity):
        if not isinstance(entity, dict):
            return
        name = entity.get("name")
        if entity.get("private_attribute_id") is not None or not isinstance(name, str):
            return
        type_name = entity.get("private_attribute_entity_type_name")
        if type_name == "ImportedSurface":
            entity["private_attribute_id"] = f"{name}_defaultBody"
        elif type_name in _LEGACY_OPTIONAL_ID_TYPE_NAMES:
            entity["private_attribute_id"] = name

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in entity_list_keys and isinstance(value, list):
                    for item in value:
                        backfill(item)
                elif key in grouped_list_keys and isinstance(value, list):
                    for group in value:
                        if isinstance(group, list):
                            for item in group:
                                backfill(item)
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(params_as_dict)
    return params_as_dict


# The GhostSurface placeholder type was retired: ghost names map deterministically
# to the real metadata-backed ghost types (a closed vocabulary — the pipeline
# dictates every ghost name, see flow360scripts/preprocess/ghostEntitiesFromMetadata.py).
_GHOST_ENTITY_TYPE_NAMES = frozenset({"GhostSphere", "GhostCircularPlane", "WindTunnelGhostSurface"})

_DRAFT_ENTITY_TYPE_NAMES = frozenset(
    {
        "AxisymmetricBody",
        "Box",
        "VoxelGrid",
        "Cylinder",
        "Sphere",
        "Point",
        "PointArray",
        "PointArray2D",
        "Slice",
        "CustomVolume",
        "SeedpointVolume",
    }
)


def _resolve_ghost_surface_type_name(name):
    """Map a legacy GhostSurface token name to its real ghost type name."""
    if name == "farfield":
        return "GhostSphere"
    if name in ("symmetric", "symmetric-1", "symmetric-2"):
        return "GhostCircularPlane"
    if isinstance(name, str) and name.startswith("windTunnel"):
        return "WindTunnelGhostSurface"
    raise ValueError(
        f"[Updater] Legacy GhostSurface token with unknown ghost name '{name}' — "
        "cannot map it to a real ghost entity type."
    )


def _convert_ghost_surface_tokens(params_as_dict):
    """Rewrite every legacy ``GhostSurface`` dict to its real ghost type.

    GhostSurface was a Python-API-only placeholder carrying just (name, id);
    the rewrite keeps those fields and swaps the discriminator, producing a
    partial instance of the metadata-backed type (geometry fields are Optional
    by FE convention). Covers stored_entities blobs and GhostSurfacePair
    members alike — anything GhostSurface-typed anywhere in the params.
    """

    def visit(node):
        if isinstance(node, dict):
            if node.get("private_attribute_entity_type_name") == "GhostSurface":
                node["private_attribute_entity_type_name"] = _resolve_ghost_surface_type_name(node.get("name"))
                # Legacy GhostSurface predates required ids; ghost ids are
                # name-determined, so stamp the id wherever it is missing
                # (covers GhostSurfacePair members the list backfill never sees).
                if not node.get("private_attribute_id"):
                    node["private_attribute_id"] = node.get("name")
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(params_as_dict)
    return params_as_dict


def _convert_wall_glob_to_selector(params_as_dict):
    """Convert the legacy ``Surface(name="*")`` glob-hack to a tokenized selector.

    Old default-shaped files encoded "all surfaces" as a fake Surface entity
    named ``*``. Patterns are selectors, never entities: drop the fake entity
    and register an equivalent ``SurfaceSelector`` (token in the EntityList,
    definition in ``asset_cache.used_selectors`` — the FE-ready wire form).
    """

    def make_selector_definition(node):
        selector_name = "Wall surfaces" if node.get("type") == "Wall" else "All surfaces"
        return {
            "target_class": "Surface",
            "selector_id": generate_uuid(),
            "name": selector_name,
            "logic": "AND",
            "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
        }

    def get_used_selectors():
        asset_cache = params_as_dict.setdefault("private_attribute_asset_cache", {})
        used_selectors = asset_cache.get("used_selectors")
        if not isinstance(used_selectors, list):
            used_selectors = []
            asset_cache["used_selectors"] = used_selectors
        return used_selectors

    def is_glob_surface(item):
        return (
            isinstance(item, dict)
            and item.get("private_attribute_entity_type_name") == "Surface"
            and item.get("name") == "*"
        )

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "private_attribute_asset_cache":
                    continue
                if isinstance(value, dict):
                    stored_entities = value.get("stored_entities")
                    if isinstance(stored_entities, list) and any(is_glob_surface(item) for item in stored_entities):
                        selector_definition = make_selector_definition(node)
                        get_used_selectors().append(selector_definition)
                        value["stored_entities"] = [item for item in stored_entities if not is_glob_surface(item)]
                        value.setdefault("selectors", []).append(selector_definition["selector_id"])
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(params_as_dict)
    return params_as_dict


def _merge_divergent_zone_fields(inline_dict, entity_info_entry):
    """Targeted divergence merge for user-assignable physics fields.

    Only ``GenericVolume``/``CustomVolume`` carry fields a user can set on the
    assignment copy (``axis``/``axes``/``center``). Inline non-null + entity_info
    null → copy over; both non-null and different → inline wins (the translator
    historically consumed the inline blob, so inline is what the solver ran).
    Every other type/field: no content comparison at all.
    """
    if inline_dict.get("private_attribute_entity_type_name") not in ("GenericVolume", "CustomVolume"):
        return
    for field in ("axis", "axes", "center"):
        inline_value = inline_dict.get(field)
        if inline_value is not None and inline_value != entity_info_entry.get(field):
            entity_info_entry[field] = inline_value


def _inline_to_compact_refs(params_as_dict):
    """Convert inline ``stored_entities`` blobs to compact reference groups.

    Every EntityList assignment outside the asset cache becomes ordered
    adjacent-run groups ``{"type": ..., "ids": [...]}`` resolving against
    ``project_entity_info``:

    - ``(type, id)`` match → reference (after the targeted divergence merge).
    - id miss → unique ``(type, name)`` match → reference to the entity_info
      entry; multiple name matches are disambiguated by content equality
      (ignoring id) — still ambiguous → hard error.
    - Ghost types reference by ``(type, id)`` unconditionally (self-identifying
      names; an unresolvable ghost fails loudly at load).
    - Truly absent draft-type entities are backfilled into ``draft_entities``;
      a persistent-type entity absent from entity_info stays inline (Mode-1
      semantics) until the serializer flip makes inline dicts a hard error.

    Files without ``project_entity_info`` (pure-local saves) are left inline:
    there is no definitions store to reference into, and the loader keeps
    Mode-1 semantics for inline dicts until the serializer flip.
    """
    asset_cache = params_as_dict.get("private_attribute_asset_cache") or {}
    entity_info = asset_cache.get("project_entity_info")
    if not isinstance(entity_info, dict) or not entity_info.get("type_name"):
        return params_as_dict

    id_index = {}
    name_index = {}

    def index_entity(entity):
        if not isinstance(entity, dict):
            return
        type_name = entity.get("private_attribute_entity_type_name")
        entity_id = entity.get("private_attribute_id")
        if not isinstance(type_name, str):
            return
        if isinstance(entity_id, str):
            id_index[(type_name, entity_id)] = entity
        name = entity.get("name")
        if isinstance(name, str):
            name_index.setdefault((type_name, name), []).append(entity)

    # Assignments can only reference entities from the ACTIVE grouping (the FE
    # contract): the index mirrors what EntityRegistry.from_entity_info registers.
    from flow360_schema.models.entity_info import iter_entity_dicts_from_entity_info_dict

    for entity in iter_entity_dicts_from_entity_info_dict(entity_info, grouped="active"):
        index_entity(entity)
    for entity in asset_cache.get("imported_surfaces") or []:
        index_entity(entity)

    def content_equal(entity_a, entity_b):
        def strip(entity):
            return {key: value for key, value in entity.items() if key != "private_attribute_id"}

        return strip(entity_a) == strip(entity_b)

    def resolve_reference_id(item):
        """Return the referenced entity id for one inline entity dict."""
        type_name = item.get("private_attribute_entity_type_name")
        entity_id = item.get("private_attribute_id")
        if not isinstance(type_name, str):
            raise ValueError(f"[Updater] Inline entity without a type cannot be converted: {item}")
        if not isinstance(entity_id, str):
            if type_name not in _DRAFT_ENTITY_TYPE_NAMES:
                # Persistent-type entity without an id: left inline so the loader
                # fails loudly on the mandatory id, exactly as it does today.
                return None
            # Draft types carry uuid default factories; an id-less wire dict would
            # have received a random id at build time — stamp one now so the
            # definition can live in draft_entities.
            entity_id = generate_uuid()
            item["private_attribute_id"] = entity_id

        if type_name in _GHOST_ENTITY_TYPE_NAMES:
            return type_name, entity_id

        matched = id_index.get((type_name, entity_id))
        if matched is None:
            name_matches = name_index.get((type_name, item.get("name")), [])
            if len(name_matches) > 1:
                name_matches = [entry for entry in name_matches if content_equal(entry, item)]
            if len(name_matches) > 1:
                raise ValueError(
                    f"[Updater] Ambiguous entity_info match for inline entity "
                    f"(type: {type_name}, name: {item.get('name')})."
                )
            matched = name_matches[0] if name_matches else None

        if matched is None:
            if type_name not in _DRAFT_ENTITY_TYPE_NAMES:
                # Persistent-type entity absent from entity_info: draft_entities
                # cannot hold it, so it stays inline with master Mode-1 semantics
                # until the serializer flip makes inline dicts a hard error.
                return None
            entity_info.setdefault("draft_entities", []).append(item)
            index_entity(item)
            return type_name, entity_id

        _merge_divergent_zone_fields(item, matched)
        return type_name, matched.get("private_attribute_id")

    def convert_list(stored_entities):
        converted = []
        for item in stored_entities:
            if isinstance(item, dict) and set(item.keys()) == {"type", "ids"}:
                type_name, ids = item["type"], list(item["ids"])
            elif isinstance(item, dict) and "private_attribute_entity_type_name" not in item:
                # Multi-constructor shorthand ({"type_name": ..., "private_attribute_input_cache": ...}):
                # expanded by parse_model_dict during validation, so it stays inline here.
                converted.append(item)
                continue
            elif isinstance(item, dict):
                resolved = resolve_reference_id(item)
                if resolved is None:
                    converted.append(item)
                    continue
                type_name, entity_id = resolved
                ids = [entity_id]
            else:
                # Live entity instance (python-mode dump validated in-process):
                # already materialized, passes through untouched.
                converted.append(item)
                continue
            if converted and isinstance(converted[-1], dict) and converted[-1].get("type") == type_name:
                converted[-1]["ids"].extend(ids)
            else:
                converted.append({"type": type_name, "ids": ids})
        return converted

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "private_attribute_asset_cache":
                    continue
                if key == "stored_entities" and isinstance(value, list):
                    node[key] = convert_list(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(params_as_dict)
    return params_as_dict
