"""
Support class and functions for project interface.
"""

from typing import Literal, Optional, Type, TypeVar, get_args

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.physical_dimensions import Length
from flow360_schema.models.asset_cache import AssetCache
from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup
from flow360_schema.models.entities.surface_entities import ImportedSurface, Surface
from flow360_schema.models.entity_info import (
    DraftEntityTypes,
    EntityInfoModel,
    GeometryEntityInfo,
)
from flow360_schema.models.reference_geometry import ProjectedArea
from flow360_schema.models.simulation.outputs.outputs import (
    SurfaceIntegralOutput,
    SurfaceOutput,
)
from flow360_schema.models.simulation.services_utils import (
    strip_selector_matches_and_broken_entities_inplace,
)
from flow360_schema.models.simulation.simulation_params import SimulationParams
from pydantic import ValidationError

from flow360.component.simulation import services
from flow360.component.simulation.draft_context import get_active_draft
from flow360.component.simulation.measurement.projection import _projected_area
from flow360.component.simulation.user_code.core.types import save_user_variables
from flow360.component.simulation.warning_bypass import (
    LENGTH_SCALE_MISMATCH,
    is_warning_bypassed,
)
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.exceptions import (
    Flow360ConfigurationError,
    Flow360RuntimeError,
    Flow360ValueError,
)
from flow360.log import log

T = TypeVar("T", bound=Flow360BaseModel)


def _apply_geometry_grouping_overrides(
    entity_info: GeometryEntityInfo,
    face_grouping: Optional[str],
    edge_grouping: Optional[str],
) -> dict[str, Optional[str]]:
    """Apply explicit face/edge grouping overrides onto geometry entity info."""

    # >>> 1. Select groupings to use, either from overrides or entity_info defaults.

    def _select_tag(new_tag, default_tag, kind):
        if new_tag is not None:
            tag = new_tag
        else:
            log.debug(
                f"No {kind} grouping specified when creating draft; "
                f"using {kind} grouping: {default_tag} from `new_run_from`."
            )
            tag = default_tag
        return tag

    face_tag = _select_tag(face_grouping, entity_info.face_group_tag, "face")
    edge_tag = _select_tag(edge_grouping, entity_info.edge_group_tag, "edge")
    body_group_tag = (
        "groupByFile"
        if "groupByFile" in entity_info.body_attribute_names
        else entity_info.body_group_tag
    )

    # >>> 2. Validate groupings
    def _validate_tag(tag, available: list[str], kind: str) -> str:
        if not available:
            raise Flow360ValueError(
                f"Unexpected {kind} grouping error: "
                f"The activated geometries in the draft do not have any {kind} grouping in common."
            )
        if tag not in available:
            raise Flow360ValueError(
                f"The current {kind} grouping '{tag}' is not valid in the geometry. "
                f"Please specify a valid {kind} grouping via `fl.create_draft({kind}_grouping=...)`. "
                f"Available tags: {available}."
            )
        return tag

    face_tag = _validate_tag(face_tag, entity_info.face_attribute_names, "face")
    # face_tag must be specified either from override or entity_info default
    assert face_tag is not None, log.debug(
        "[Internal] Default face grouping should be set, face tag to be applied: ", face_tag
    )
    entity_info._group_entity_by_tag("face", face_tag)  # pylint:disable=protected-access
    # edge_tag can be None if the geometry asset created with surface mesh
    if edge_grouping is not None and entity_info.edge_attribute_names:
        edge_tag = _validate_tag(edge_tag, entity_info.edge_attribute_names, "edge")
        entity_info._group_entity_by_tag("edge", edge_tag)  # pylint:disable=protected-access

    entity_info._group_entity_by_tag("body", body_group_tag)  # pylint:disable=protected-access

    return {
        "face": entity_info.face_group_tag,
        "edge": entity_info.edge_group_tag,
        "body": entity_info.body_group_tag,  # Not used since customized body grouping is not supported yet
    }


def load_status_from_asset(
    *,
    asset: AssetBase,
    status_class: Type[T],
    cache_key: str,
) -> Optional[T]:
    """
    Retrieve a cached status object from an asset's simulation metadata.

    Parameters
    ----------
    asset : AssetBase
        Asset that owns the cache.
    status_class : Type[T]
        Target status model to deserialize.
    cache_key : str
        Cache key name.

    Returns
    -------
    Optional[T]
        Parsed status instance or None when not present.
    """

    # pylint: disable=protected-access
    if hasattr(asset, "_simulation_dict_cache_for_local_mode"):
        simulation_dict = asset._simulation_dict_cache_for_local_mode
    else:
        simulation_dict = AssetBase._get_simulation_json(asset=asset, clean_front_end_keys=True)

    status_dict = simulation_dict.get("private_attribute_asset_cache", {}).get(cache_key, None)
    if status_dict is None:
        return None

    try:
        return status_class.deserialize(status_dict)
    except ValidationError as exc:  # pragma: no cover - raises immediately
        status_name = cache_key.replace("_", " ")
        raise Flow360RuntimeError(
            f"[Internal] Failed to parse stored {status_name} for {asset.__class__.__name__}. Error: {exc}",
        ) from exc


def apply_and_inform_grouping_selections(
    *,
    entity_info,
    face_grouping: Optional[str],
    edge_grouping: Optional[str],
    new_run_from_geometry: bool,
) -> None:
    """
    Apply and emit logging messages describing which geometry grouping tags will be used.

    Highlights legacy registry-derived tags so users can migrate to explicit DraftContext
    overrides via create_draft().
    """

    if not isinstance(entity_info, GeometryEntityInfo):
        if face_grouping is None and edge_grouping is None:
            return
        log.warning(
            "Ignoring face/edge grouping (%s/%s): only geometry assets support face/edge grouping.",
            face_grouping,
            edge_grouping,
        )
        return

    applied_grouping = _apply_geometry_grouping_overrides(entity_info, face_grouping, edge_grouping)

    # 1. Print out the grouping used for user's convenience.

    log.info(
        "Creating draft with geometry grouping:\n  faces: %s\n  edges: %s\n  bodies: %s\n",
        applied_grouping.get("face"),
        applied_grouping.get("edge"),
        applied_grouping.get("body"),
    )

    missing_groupings = []
    if face_grouping is None:
        missing_groupings.append("face_grouping")
    if edge_grouping is None and entity_info.edge_attribute_names:
        missing_groupings.append("edge_grouping")

    if missing_groupings and new_run_from_geometry:
        # We had to use legacy grouping from asset metadata.
        # Warning is only required if starting from a geometry resource, otherwise we should use the
        # grouping encoded in the non-geometry resource.
        log.warning(
            "%s not specified when creating draft and therefore come from geometry asset object. "
            "This support will be deprecated in the future. Please specify all groupings during the draft creation"
            " (`create_draft(face_grouping='...', edge_grouping='...', ...)`) instead.",
            " and ".join(missing_groupings),
        )


def _set_up_params_non_persistent_entity_info(entity_info, params: SimulationParams):
    """
    Setting up non-persistent entities (AKA draft entities) in params.
    Add the ones used to the entity info.

    LEGACY: This function is used for the legacy workflow (without DraftContext).
    For DraftContext workflow, use _merge_draft_entities_from_params() instead.
    """

    entity_registry = params.used_entity_registry
    existing_entity_keys = {
        (entity.private_attribute_entity_type_name, entity.private_attribute_id)
        for entity in entity_info.draft_entities
    }
    # Creating draft entities: derive classes from DraftEntityTypes to avoid duplication
    # DraftEntityTypes is Annotated[Union[...], Field(...)], so the Union is the first arg
    draft_type_union = get_args(DraftEntityTypes)[0]
    draft_type_list = get_args(draft_type_union)
    for draft_type in draft_type_list:
        draft_entities = list(entity_registry.view(draft_type))
        for draft_entity in draft_entities:
            entity_key = (
                draft_entity.private_attribute_entity_type_name,
                draft_entity.private_attribute_id,
            )
            if entity_key in existing_entity_keys:
                continue
            entity_info.draft_entities.append(draft_entity)
            existing_entity_keys.add(entity_key)
    return entity_info


def _set_up_params_imported_surfaces(params: SimulationParams):
    """
    Setting up imported_surfaces in params.
    Add the ones used to the outputs.
    """

    if not params.outputs:
        return params

    imported_surfaces = {}

    for output in params.outputs:
        if not isinstance(output, (SurfaceOutput, SurfaceIntegralOutput)):
            continue
        for surface in output.entities.stored_entities:
            if isinstance(surface, ImportedSurface) and surface.name not in imported_surfaces:
                imported_surfaces[surface.name] = surface

    params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
        "imported_surfaces", list(imported_surfaces.values())
    )

    return params


def _merge_draft_entities_from_params(
    entity_info: EntityInfoModel,
    params: SimulationParams,
) -> EntityInfoModel:
    """
    Collect draft entities from params.used_entity_registry and merge into entity_info.

    This function implements the merging logic for the DraftContext workflow:
    - If a draft entity already exists in entity_info (by ID), use entity_info version (source of truth)
    - If a draft entity is new (not in entity_info), add it from params

    This ensures that:
    1. Entities managed by DraftContext retain their modifications
    2. New entities created by the user during simulation setup are captured

    Parameters:
        entity_info: The entity_info to merge into (typically from DraftContext)
        params: The SimulationParams containing used_entity_registry

    Returns:
        EntityInfoModel: The updated entity_info with merged draft entities
    """
    used_registry = params.used_entity_registry

    # Get all draft entity types from the DraftEntityTypes annotation
    draft_type_union = get_args(DraftEntityTypes)[0]
    draft_type_list = get_args(draft_type_union)

    # Build a set of IDs already in entity_info for quick lookup (Draft entities have unique UUIDs)
    existing_ids = {e.private_attribute_id for e in entity_info.draft_entities}

    for draft_type in draft_type_list:
        draft_entities_used = list(used_registry.view(draft_type))
        for draft_entity in draft_entities_used:
            # Only add if not already in entity_info (by ID)
            # If already present, entity_info version is source of truth - keep it as is
            if draft_entity.private_attribute_id not in existing_ids:
                entity_info.draft_entities.append(draft_entity)
                existing_ids.add(draft_entity.private_attribute_id)

    return entity_info


def _update_entity_grouping_tags(entity_info, params: SimulationParams) -> EntityInfoModel:
    """
    Update the entity grouping tags in params to resolve possible conflicts
    between the SimulationParams and the root asset.This
    """

    def _get_used_tags(model: Flow360BaseModel, target_entity_type, used_tags: set):
        for field in model.__dict__.values():
            # Skip the AssetCache since the asset cache is exactly what we want to update later.

            if isinstance(field, AssetCache):
                continue

            if isinstance(field, target_entity_type):
                used_tags.add(field.private_attribute_tag_key)

            if isinstance(field, EntityList):
                for entity in field.stored_entities:
                    if isinstance(entity, target_entity_type):
                        used_tags.add(entity.private_attribute_tag_key)

            elif isinstance(field, (list, tuple)):
                for item in field:
                    if isinstance(item, target_entity_type):
                        used_tags.add(item.private_attribute_tag_key)
                    elif isinstance(item, Flow360BaseModel):
                        _get_used_tags(item, target_entity_type, used_tags)

            elif isinstance(field, Flow360BaseModel):
                _get_used_tags(field, target_entity_type, used_tags)

    if entity_info.type_name != "GeometryEntityInfo":
        return entity_info
    # pylint: disable=protected-access
    entity_types = [
        (Surface, "face_group_tag"),
    ]

    if entity_info.all_edge_ids:
        entity_types.append((Edge, "edge_group_tag"))

    if entity_info.all_body_ids:
        entity_types.append((GeometryBodyGroup, "body_group_tag"))

    for entity_type, entity_grouping_tags in entity_types:
        used_tags = set()
        _get_used_tags(params, entity_type, used_tags)

        if None in used_tags:
            used_tags.remove(None)

        used_tags = sorted(list(used_tags))
        current_tag = getattr(entity_info, entity_grouping_tags)

        # If explicit entities were stripped (e.g. selector-only usage), we may have no tags
        # discoverable from the params object. In that case, fall back to the grouping tags
        # already recorded in the params asset cache.
        if not used_tags:
            asset_cache = getattr(params, "private_attribute_asset_cache", None)
            cached_entity_info = getattr(asset_cache, "project_entity_info", None)
            cached_tag = (
                getattr(cached_entity_info, entity_grouping_tags, None)
                if cached_entity_info is not None
                and getattr(cached_entity_info, "type_name", None) == "GeometryEntityInfo"
                else None
            )
            if cached_tag is not None:
                used_tags = [cached_tag]

        if len(used_tags) == 1 and current_tag != used_tags[0]:
            log.warning(
                f"Inconsistent grouping of {entity_type.__name__} between the geometry object ({current_tag})"
                f" and SimulationParams ({used_tags[0]}). "
                "Ignoring the geometry object and using the one in the SimulationParams."
            )
            entity_info._force_set_attr(entity_grouping_tags, used_tags[0])

        if len(used_tags) > 1:
            raise Flow360ConfigurationError(
                f"Multiple entity ({entity_type.__name__}) grouping tags found "
                f"in the SimulationParams ({used_tags})."
            )

    return entity_info


def _set_up_default_geometry_accuracy(
    root_asset,
    params: SimulationParams,
    use_geometry_AI: bool,  # pylint: disable=invalid-name
):
    """
    Set up the default geometry accuracy in params if not set by the user.
    """
    if not use_geometry_AI:
        return params
    if root_asset.default_settings.get("geometry_accuracy") is None:
        return params
    if not params.meshing.defaults.geometry_accuracy:
        params.meshing.defaults.geometry_accuracy = root_asset.default_settings["geometry_accuracy"]
        log.info(
            "Setting up default geometry accuracy for GAI as: %s",
            str(params.meshing.defaults.geometry_accuracy),
        )
    return params


def _set_up_default_reference_geometry(params: SimulationParams, length_unit: Length.Float64):
    """
    Setting up the default reference geometry if not provided in params.
    Ensure the simulation.json contains the default settings other than None.
    """
    # pylint: disable=protected-access
    default_reference_geometry = services._get_default_reference_geometry(length_unit)
    if params.reference_geometry is None:
        params.reference_geometry = default_reference_geometry
        return params

    for field in params.reference_geometry.__class__.model_fields:
        if getattr(params.reference_geometry, field) is None:
            setattr(
                params.reference_geometry,
                field,
                getattr(default_reference_geometry, field),
            )

    return params


_HALF_BODY_DOMAIN_CLIP = {"half_body_negative_y": "-Y", "half_body_positive_y": "+Y"}


def _meshed_half_space(params: SimulationParams):
    """Return the half space the meshed domain keeps, with the setting that decided it.

    `domain_type` lives on the farfield base class, so reading it by attribute covers
    every farfield type, including ones added later. Any other value -- `full_body`,
    unset, quasi-3d without an explicit half body -- leaves the uploaded geometry
    untouched, so a measurement over it needs no restriction.
    """
    volume_zones = getattr(getattr(params, "meshing", None), "volume_zones", None)
    for zone in volume_zones or []:
        domain_type = getattr(zone, "domain_type", None)
        half_space = _HALF_BODY_DOMAIN_CLIP.get(domain_type)
        if half_space is not None:
            return half_space, domain_type
    return None, None


def _compute_projected_reference_area(params: SimulationParams, active_draft) -> None:
    """Measure a ``ProjectedArea`` reference area and store the result on it in place.

    Does nothing unless ``reference_geometry.area`` is a ``ProjectedArea``; a numeric or
    expression area is already final. Measures over the half body the meshing settings
    actually produce, so the stored value describes the model the solver will see.

    Runs at submission because it needs both finalized meshing settings and a live draft
    holding the geometry tessellation.
    """
    area = getattr(params.reference_geometry, "area", None)
    if not isinstance(area, ProjectedArea):
        return
    if active_draft is None:
        raise Flow360RuntimeError(
            "Automatic projected reference area requires an active draft created from a "
            "Geometry resource. Submit while the draft context that owns the selected "
            "surfaces is still active."
        )

    # The reference area has to describe the model the solver actually sees. A half-body
    # domain is meshed from the geometry trimmed at Y=0, so measure the same half rather
    # than asking the user for a scale factor: deriving it here is correct whether the
    # uploaded geometry is a full model that the mesher will trim or already just one half.
    half_space, domain_type = _meshed_half_space(params)
    if half_space is not None:
        log.info(
            f"`domain_type={domain_type}` meshes the model trimmed at Y=0, so the projected "
            f"reference area is measured on the {half_space} side only."
        )

    # pylint: disable=protected-access
    computed = _projected_area(
        active_draft,
        surfaces=area.surfaces,
        direction=area.direction,
        render_quality=area.render_quality,
        clip=half_space,
    )
    area._set_computed(value=computed)
    log.info(f"Automatically computed projected reference area: {area.computed}.")


def _read_root_simulation_dict(root_asset) -> dict:
    """Return the root asset's stored simulation.json dict."""
    if hasattr(root_asset, "_simulation_dict_cache_for_local_mode"):
        # pylint:disable-next=protected-access
        return root_asset._simulation_dict_cache_for_local_mode
    return AssetBase._get_simulation_json(  # pylint:disable=protected-access
        asset=root_asset, clean_front_end_keys=True
    )


def read_root_cad_importer_version(root_asset) -> Literal["v1", "v2"]:
    """
    Return the CAD Importer version recorded on the root asset, defaulting to
    "v1" when the root is an older geometry whose simulation.json predates the
    field. Used to keep an imported geometry dependency on the same importer as
    the project root.
    """
    root_engine = (
        _read_root_simulation_dict(root_asset).get("private_attribute_asset_cache") or {}
    ).get("cad_importer_version")
    return root_engine or "v1"


def _enforce_inheritance_of_root_asset_workflow_settings(root_asset, params: SimulationParams):
    """
    Enforce (mostly workflow) settings from the root asset to avoid drifting.
    """
    # FXC-3289: propagate the CAD Importer version from the root asset's stored
    # asset_cache. The geometry resource fixes this at upload time;
    # ensure_cad_importer_compatible_with_mesher reads it on the surface mesh
    # submission to reject v2 + beta mesher before the job dispatches.
    _root_engine = (
        _read_root_simulation_dict(root_asset).get("private_attribute_asset_cache") or {}
    ).get("cad_importer_version")
    if _root_engine is not None:
        params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
            "cad_importer_version", _root_engine
        )
    return params


def ensure_cad_importer_compatible_with_mesher(params: SimulationParams):
    """
    Reject CAD Importer v2 combined with the standalone beta in-house surface
    mesher (beta enabled without Geometry AI) at submission time.

    CAD Importer v2 (HOOPS only) never emits the .egads file v1 produces. The
    beta in-house surface mesher on its own reads that EGADS face partition
    directly and would crash mid-run when no .egads is available. The default
    surface mesher and Geometry AI are both supported on v2 -- Geometry AI's
    surface mesher re-tessellates the stamped v2 STEP through the HOOPS importer
    instead of the EGADS partition.

    Call after the CAD Importer version is inherited from the root asset.
    """
    asset_cache = params.private_attribute_asset_cache
    if asset_cache.cad_importer_version != "v2":
        return
    if asset_cache.use_inhouse_mesher and not asset_cache.use_geometry_AI:
        raise Flow360ValueError(
            "The beta in-house surface mesher (without Geometry AI) requires CAD Importer V1. "
            "Re-upload this project with CAD Importer V1, or enable Geometry AI."
        )


def set_up_params_for_uploading(  # pylint: disable=too-many-arguments
    root_asset,
    length_unit: Length.Float64,
    params: SimulationParams,
    use_beta_mesher: bool,
    use_geometry_AI: bool,  # pylint: disable=invalid-name
) -> SimulationParams:
    """
    Set up params before submitting the draft.

    Parameters:
        root_asset: The root asset (Geometry, SurfaceMesh, or VolumeMesh).
        length_unit: The project length unit.
        params: The SimulationParams to set up.
        use_beta_mesher: Whether to use the beta mesher.
        use_geometry_AI: Whether to use Geometry AI.
    """

    params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
        "project_length_unit", length_unit
    )

    params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
        "use_inhouse_mesher",
        use_beta_mesher if use_beta_mesher else False,
    )

    params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
        "use_geometry_AI",
        use_geometry_AI if use_geometry_AI else False,
    )

    _enforce_inheritance_of_root_asset_workflow_settings(root_asset, params)

    active_draft = get_active_draft()

    if active_draft is not None:
        # New DraftContext workflow: use draft's entity_info as source of truth
        # Merge draft entities from params.used_entity_registry into draft_entity_info
        # pylint: disable=protected-access
        entity_info = _merge_draft_entities_from_params(active_draft._entity_info, params)

        # Update entity grouping tags if needed
        # (back compatibility, since the grouping should already have been captured in the draft_entity_info)
        entity_info = _update_entity_grouping_tags(entity_info, params)

        mirror_status = active_draft.mirror._mirror_status
        if not mirror_status.is_empty():
            params.private_attribute_asset_cache._force_set_attr("mirror_status", mirror_status)
        else:
            params.private_attribute_asset_cache._force_set_attr("mirror_status", None)
        params.private_attribute_asset_cache._force_set_attr(
            "coordinate_system_status",
            active_draft.coordinate_systems._to_status(),
        )
    else:
        # Legacy workflow (without DraftContext): use root_asset.entity_info
        # User may have made modifications to the entities which is recorded in asset's entity registry
        # We need to reflect these changes.
        entity_info = root_asset.entity_info
        entity_info.update_persistent_entities(asset_entity_registry=params.used_entity_registry)

        # Check if there are any new draft entities that have been added in the params by the user
        entity_info = _set_up_params_non_persistent_entity_info(entity_info, params)

        # If the customer just load the param without re-specify the same set of entity grouping tags,
        # we need to update the entity grouping tags to the ones in the SimulationParams.
        entity_info = _update_entity_grouping_tags(entity_info, params)

    # At this point the draft entity info has replaced the SimulationParams's entity info.
    # So the validation afterwards does not require the access to the draft entity info anymore.
    params.private_attribute_asset_cache._force_set_attr(  # pylint:disable=protected-access
        "project_entity_info", entity_info
    )
    params = _set_up_default_geometry_accuracy(root_asset, params, use_geometry_AI)

    params = _set_up_default_reference_geometry(params, length_unit)

    _compute_projected_reference_area(params, active_draft)

    # Convert all reference of UserVariables to VariableToken
    params = save_user_variables(params)

    # Set up imported surfaces in params
    params = _set_up_params_imported_surfaces(params)

    # Strip selector-matched entities from stored_entities before upload so that hand-picked
    # entities remain distinguishable on the UI side.
    strip_selector_matches_and_broken_entities_inplace(params)

    return params


# Ratio = largest-bounding-box-dimension / requested setting value. Mirrors the WebUI thresholds
# (flow360-ui-next .../hooks/use-meshing-warnings.ts) so both paths flag the same cases.
# A "too fine" setting (large ratio) usually signals an incorrect CAD length unit/scale and wasted
# compute -> confirm gate. A "too coarse" geometry_accuracy (small ratio) is only advisory -> soft warn.
MAX_EDGE_LENGTH_TOO_FINE_RATIO = 500  # surface_max_edge_length < bbox_max / 500
GEOMETRY_ACCURACY_TOO_FINE_RATIO = 100_000  # geometry_accuracy < bbox_max / 100000
GEOMETRY_ACCURACY_TOO_COARSE_RATIO = 100  # geometry_accuracy > bbox_max / 100


def _collect_length_settings(meshing) -> list:
    """(label, value) pairs of mesh length settings to sanity-check against the bounding box."""
    defaults = getattr(meshing, "defaults", None)
    refinements = getattr(meshing, "refinements", None) or []
    settings = [
        ("max_edge_length", getattr(defaults, "surface_max_edge_length", None)),
        ("geometry_accuracy", getattr(defaults, "geometry_accuracy", None)),
    ]
    for refinement in refinements:
        settings.append(("max_edge_length", getattr(refinement, "max_edge_length", None)))
        settings.append(("geometry_accuracy", getattr(refinement, "geometry_accuracy", None)))
    return [(label, value) for label, value in settings if value is not None]


def _confirm_length_scale_proceed(warning_message: str) -> bool:
    """Interactively confirm proceeding past a likely length-scale mismatch.

    Returns True only on an explicit ``y``. A declined prompt or a non-interactive
    session (no stdin) returns False.
    """
    log.warning(warning_message)
    print("Proceed with submission anyway? (y/n): ")
    while True:
        try:
            answer = input().strip().lower()
        except EOFError:
            return False
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Enter a valid value (y/n): ")


def enforce_length_scale_sanity(params: SimulationParams) -> bool:
    """Decide whether to proceed when a requested mesh length setting is implausibly fine.

    Compares the geometry's largest bounding-box dimension against each requested length setting
    (``max_edge_length``, ``geometry_accuracy``), mirroring the WebUI thresholds. A "too fine"
    setting usually means the geometry was uploaded with the wrong length unit/scale (e.g. an inches
    model treated as meters), which would launch a needlessly huge meshing job — the user is asked
    to confirm. A "too coarse" ``geometry_accuracy`` is only an advisory soft warning and never
    blocks. Pre-acknowledge the confirm prompt — e.g. in a batch loop — via
    ``with warning_bypass("potential_length_scale_mismatch"):``.

    Returns ``True`` to proceed (no mismatch, acknowledged, or confirmed), ``False`` when a
    mismatch was declined or could not be confirmed (non-interactive session). The caller decides
    whether to raise or return based on its ``raise_on_error`` policy.
    """
    if is_warning_bypassed(LENGTH_SCALE_MISMATCH):
        return True

    asset_cache = params.private_attribute_asset_cache
    bounding_box = getattr(asset_cache.project_entity_info, "global_bounding_box", None)
    project_length_unit = asset_cache.project_length_unit
    if bounding_box is None or project_length_unit is None or params.meshing is None:
        return True

    bounding_box_largest_dimension = bounding_box.largest_dimension * project_length_unit

    # Track the worst (highest-ratio) "too fine" offender per setting -> confirm gate.
    too_fine: dict[str, str] = {}
    too_fine_ratio: dict[str, float] = {}
    too_coarse_ratio = None  # lowest-ratio geometry_accuracy -> soft warn only
    for label, value in _collect_length_settings(params.meshing):
        ratio = (bounding_box_largest_dimension / value).to_value("dimensionless")
        fine_limit = (
            MAX_EDGE_LENGTH_TOO_FINE_RATIO
            if label == "max_edge_length"
            else GEOMETRY_ACCURACY_TOO_FINE_RATIO
        )
        if ratio > fine_limit and ratio > too_fine_ratio.get(label, 0):
            too_fine_ratio[label] = ratio
            too_fine[label] = (
                f"{label} ({value}) is {ratio:.1e}x smaller than the largest dimension"
            )
        elif label == "geometry_accuracy" and ratio < GEOMETRY_ACCURACY_TOO_COARSE_RATIO:
            if too_coarse_ratio is None or ratio < too_coarse_ratio:
                too_coarse_ratio = ratio

    if too_coarse_ratio is not None:
        log.warning(
            f"geometry_accuracy is coarse relative to the geometry's largest bounding-box dimension "
            f"({bounding_box_largest_dimension}); the surface mesh may under-resolve the geometry."
        )

    if not too_fine:
        return True

    warning_message = (
        f"The requested mesh resolution is implausibly fine for the geometry's largest bounding-box "
        f"dimension ({bounding_box_largest_dimension}): " + "; ".join(too_fine.values()) + ". This "
        "usually means the geometry was uploaded with an incorrect length unit or scale and would "
        "launch an unnecessarily large meshing job. Please verify the geometry's length unit or your "
        f"mesh settings. To proceed without prompting (e.g. in a batch loop), wrap the run in "
        f'`with warning_bypass("{LENGTH_SCALE_MISMATCH}"):`.'
    )
    return _confirm_length_scale_proceed(warning_message)
