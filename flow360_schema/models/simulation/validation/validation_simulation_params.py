"""
validation for SimulationParams
"""

from typing import get_args

from flow360_schema.framework.entity.coordinate_system_state import CoordinateSystemState
from flow360_schema.framework.entity.entity_operation import (
    _extract_scale_from_matrix,
    _is_uniform_scale,
)
from flow360_schema.framework.entity.entity_utils import is_exact_instance
from flow360_schema.models.entities.volume_entities import (
    AxisymmetricBody,
    CustomVolume,
    Cylinder,
    SeedpointVolume,
    Sphere,
)
from flow360_schema.models.entity_info import (
    GeometryEntityInfo,
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360_schema.models.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
)
from flow360_schema.models.simulation.meshing_param.volume_params import (
    CustomZones,
    RotationCylinder,
    RotationSphere,
    RotationVolume,
    WindTunnelFarfield,
)
from flow360_schema.models.simulation.models.material import Air, Gas
from flow360_schema.models.simulation.models.solver_numerics import (
    KrylovLinearSolver,
    NoneSolver,
    RoeFlux,
)
from flow360_schema.models.simulation.models.surface_models import (
    Inflow,
    Outflow,
    PorousJump,
    SurfaceModelTypes,
    Wall,
)
from flow360_schema.models.simulation.models.volume_models import (
    ActuatorDisk,
    Fluid,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360_schema.models.simulation.outputs.outputs import (
    IsosurfaceOutput,
    ProbeOutput,
    SliceOutput,
    SurfaceOutput,
    TimeAverageIsosurfaceOutput,
    TimeAverageOutputTypes,
    TimeAverageProbeOutput,
    TimeAverageSurfaceOutput,
    VolumeOutput,
)
from flow360_schema.models.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360_schema.models.simulation.validation.validation_context import (
    ALL,
    CASE,
    ParamsValidationInfo,
    add_validation_warning,
    get_validation_levels,
)
from flow360_schema.models.simulation.validation.validation_utils import (
    EntityUsageMap,
    find_user_symmetry_surfaces,
)


def _populate_validated_field_to_validation_context(v, param_info, attribute_name):
    """Populate validated objects to validation context.

    Sets the attribute to an empty dict {} when v is None or empty list,
    distinguishing successful validation with no items from validation errors
    (which leave the attribute as None).
    """
    if v is None or len(v) == 0:
        setattr(param_info, attribute_name, {})
        return v
    setattr(
        param_info,
        attribute_name,
        {
            obj.private_attribute_id: obj
            for obj in v
            if hasattr(obj, "private_attribute_id") and obj.private_attribute_id is not None
        },
    )
    return v


def _check_consistency_wall_function_and_surface_output(v):
    models = v.models

    if models:
        has_wall_function_model = False
        for model in models:
            if isinstance(model, Wall) and model.use_wall_function is not None:
                has_wall_function_model = True
                break

        if has_wall_function_model:
            return v

    outputs = v.outputs

    if outputs is None:
        return v

    for output in outputs:
        if isinstance(output, SurfaceOutput):
            if "wallFunctionMetric" in output.output_fields.items:
                raise ValueError(
                    "To use 'wallFunctionMetric' for output specify a Wall model with use_wall_function=true. "
                )

    return v


def _check_duplicate_entities_in_models(params, param_info: ParamsValidationInfo):
    if not params.models:
        return params

    models = params.models
    usage = EntityUsageMap()

    for model in models:
        if hasattr(model, "entities") and model.entities is not None:
            expanded_entities = param_info.expand_entity_list(model.entities)
            for entity in expanded_entities:
                usage.add_entity_usage(entity, model.type)

    error_msg = ""
    for entity_type, entity_model_map in usage.dict_entity.items():
        for entity_info in entity_model_map.values():
            if len(entity_info["model_list"]) > 1:
                model_set = set(entity_info["model_list"])
                model_string = ", ".join(f"`{x}`" for x in sorted(model_set))
                model_string += " models.\n" if len(model_set) > 1 else " model.\n"
                error_msg += (
                    f"{entity_type} entity `{entity_info['entity_name']}` "
                    + f"appears multiple times in {model_string}"
                )

    if error_msg:
        raise ValueError(error_msg)

    return params


def _check_low_mach_preconditioner_output(v):
    models = v.models

    if models:
        has_low_mach_preconditioner = False
        for model in models:
            if isinstance(model, Fluid) and model.navier_stokes_solver:
                riemann_solver = model.navier_stokes_solver.riemann_solver
                if isinstance(riemann_solver, RoeFlux) and riemann_solver.low_mach_preconditioner:
                    has_low_mach_preconditioner = True
                    break

        if has_low_mach_preconditioner:
            return v

    outputs = v.outputs

    if not outputs:
        return v

    for output in outputs:
        if not hasattr(output, "output_fields"):
            continue
        if "lowMachPreconditionerSensor" in output.output_fields.items:
            raise ValueError(
                "Low-Mach preconditioner output requested, but low_mach_preconditioner is not enabled. "
                "You can enable it via "
                "model.navier_stokes_solver.riemann_solver = RoeFlux(low_mach_preconditioner=True) "
                "for a Fluid model in the models field of the simulation object."
            )

    return v


def _check_numerical_dissipation_factor_output(v):
    models = v.models

    if models:
        low_dissipation_enabled = False
        for model in models:
            if isinstance(model, Fluid) and model.navier_stokes_solver:
                riemann_solver = model.navier_stokes_solver.riemann_solver
                if not isinstance(riemann_solver, RoeFlux):
                    continue
                low_dissipation_flag = int(round(1.0 / riemann_solver.numerical_dissipation_factor)) - 1
                if low_dissipation_flag != 0:
                    low_dissipation_enabled = True
                    break

        if low_dissipation_enabled:
            return v

    outputs = v.outputs

    if not outputs:
        return v

    for output in outputs:
        if not hasattr(output, "output_fields"):
            continue
        if "numericalDissipationFactor" in output.output_fields.items:
            raise ValueError(
                "Numerical dissipation factor output requested, but low dissipation mode is not enabled. "
                "You can enable it via "
                "model.navier_stokes_solver.riemann_solver = RoeFlux(numerical_dissipation_factor=0.2) "
                "for a Fluid model in the models field of the simulation object."
            )

    return v


def _check_consistency_hybrid_model_output(v):
    """
    Validate that hybridModel output fields are only requested when hybrid RANS-LES is enabled.

    We scan all outputs rather than checking isinstance() for specific types, because the real
    invariant is field-name-based: the solver sets outputDetachedEddySimulationFields from
    outputConfigsRequestField() which scans all output sections for the field name.
    """
    model_type = None
    models = v.models

    run_hybrid_model = False

    if models:
        for model in models:
            if isinstance(model, Fluid):
                turbulence_model_solver = model.turbulence_model_solver
                if (
                    not isinstance(turbulence_model_solver, NoneSolver)
                    and turbulence_model_solver.hybrid_model is not None
                ):
                    model_type = turbulence_model_solver.type_name
                    run_hybrid_model = True
                    break

    outputs = v.outputs

    if not outputs:
        return v

    for output_index, output in enumerate(outputs):
        fields = getattr(getattr(output, "output_fields", None), "items", []) or []
        if "SpalartAllmaras_hybridModel" in fields and not (model_type == "SpalartAllmaras" and run_hybrid_model):
            raise ValueError(
                f"In `outputs`[{output_index}] {output.output_type}: SpalartAllmaras_hybridModel output can only "
                "be specified with SpalartAllmaras turbulence model and hybrid RANS-LES used."
            )
        if "kOmegaSST_hybridModel" in fields and not (model_type == "kOmegaSST" and run_hybrid_model):
            raise ValueError(
                f"In `outputs`[{output_index}] {output.output_type}: kOmegaSST_hybridModel output can only be "
                "specified with kOmegaSST turbulence model and hybrid RANS-LES used."
            )

    return v


def _check_unsteadiness_to_use_hybrid_model(v):
    models = v.models

    run_hybrid_model = False

    if models:
        for model in models:
            if isinstance(model, Fluid):
                turbulence_model_solver = model.turbulence_model_solver
                if (
                    not isinstance(turbulence_model_solver, NoneSolver)
                    and turbulence_model_solver.hybrid_model is not None
                ):
                    run_hybrid_model = True
                    break

    if run_hybrid_model and v.time_stepping is not None and isinstance(v.time_stepping, Steady):
        raise ValueError("hybrid RANS-LES model can only be used in unsteady simulations.")

    return v


def _check_hybrid_model_to_use_zonal_enforcement(v):
    models = v.models
    if not models:
        return v

    for model in models:
        if isinstance(model, Fluid):
            turbulence_model_solver = model.turbulence_model_solver
            if not isinstance(turbulence_model_solver, NoneSolver):
                if turbulence_model_solver.controls is None:
                    continue
                for index, control in enumerate(turbulence_model_solver.controls):
                    if control.enforcement is not None and turbulence_model_solver.hybrid_model is None:
                        raise ValueError(
                            f"Control region {index} must be running in hybrid RANS-LES mode to "
                            "apply zonal turbulence enforcement."
                        )

    return v


def _check_cht_solver_settings(params):
    has_heat_transfer = False

    models = params.models

    if models:
        for model in models:
            if isinstance(model, Solid):
                has_heat_transfer = True

        if has_heat_transfer is False:
            params = _validate_cht_no_heat_transfer(params)
        if has_heat_transfer is True:
            params = _validate_cht_has_heat_transfer(params)

    return params


def _validate_cht_no_heat_transfer(params):
    if params.outputs:
        for output in params.outputs:
            if isinstance(
                output,
                (SurfaceOutput, VolumeOutput, SliceOutput, ProbeOutput, TimeAverageProbeOutput, IsosurfaceOutput),
            ):
                if "residualHeatSolver" in output.output_fields.items:
                    raise ValueError(
                        f"Heat equation output variables: residualHeatSolver is requested in {output.output_type} with"
                        " no `Solid` model defined."
                    )

    return params


def _validate_cht_has_heat_transfer(params):
    time_stepping = params.time_stepping
    if isinstance(time_stepping, Unsteady):
        for model_solid in params.models:
            if isinstance(model_solid, Solid):
                if model_solid.material.specific_heat_capacity is None or model_solid.material.density is None:
                    raise ValueError(
                        "In `Solid` model -> material, both `specific_heat_capacity` and `density` "
                        "need to be specified for unsteady simulations."
                    )
                if model_solid.initial_condition is None:
                    raise ValueError(
                        "In `Solid` model, the initial condition needs to be specified for unsteady simulations."
                    )
    return params


def _collect_volume_zones(params) -> list:
    """Collect volume zones from meshing config in a schema-compatible way."""
    if isinstance(params.meshing, MeshingParams):
        return params.meshing.volume_zones or []
    if isinstance(params.meshing, ModularMeshingWorkflow):
        return params.meshing.zones or []
    return []


def _collect_asset_boundary_entities(params, param_info: ParamsValidationInfo) -> tuple[list, bool]:
    """Collect boundary entities that should be considered valid for BC completeness checks.

    This includes:
    - Persistent boundaries from asset cache
    - Farfield-related ghost boundaries, conditional on farfield method
    - Wind tunnel ghost surfaces (when applicable)

    Returns:
        tuple: (asset_boundary_entities, has_missing_private_attributes)
    """
    # IMPORTANT:
    # AssetCache.boundaries may return a direct reference into EntityInfo internal lists
    # (e.g. GeometryEntityInfo.grouped_faces[*]). Always copy before appending to avoid
    # mutating entity_info and corrupting subsequent serialization/validation.
    asset_boundary_entities = list(params.private_attribute_asset_cache.boundaries or [])
    farfield_method = params.meshing.farfield_method if params.meshing else None
    has_missing_private_attributes = False

    if not farfield_method:
        return asset_boundary_entities, has_missing_private_attributes

    # Check for legacy assets missing private_attributes before farfield-related processing
    # This check is only relevant when we need bounding box information for farfield operations
    # Only flag as legacy if ALL boundaries are missing private_attributes (not just some)
    # AND the farfield method is one that performs automatic surface deletion (auto/quasi-3d/user-defined
    # modes). For wind-tunnel farfield, missing BCs are always errors since no auto-deletion occurs
    if (
        asset_boundary_entities
        and farfield_method in ("auto", "quasi-3d", "quasi-3d-periodic", "user-defined")
        and all(getattr(item, "private_attributes", None) is None for item in asset_boundary_entities)
    ):
        has_missing_private_attributes = True

    # Filter out the ones that will be deleted by mesher (only when reliable)
    if not param_info.entity_transformation_detected and not has_missing_private_attributes:
        asset_boundary_entities = [
            item
            for item in asset_boundary_entities
            if item._will_be_deleted_by_mesher(
                entity_transformation_detected=param_info.entity_transformation_detected,
                farfield_method=farfield_method,
                global_bounding_box=param_info.global_bounding_box,
                planar_face_tolerance=param_info.planar_face_tolerance,
                half_model_symmetry_plane_center_y=param_info.half_model_symmetry_plane_center_y,
                quasi_3d_symmetry_planes_center_y=param_info.quasi_3d_symmetry_planes_center_y,
                farfield_domain_type=param_info.farfield_domain_type,
            )
            is False
        ]

    ghost_entities = getattr(params.private_attribute_asset_cache.project_entity_info, "ghost_entities", [])

    if farfield_method == "auto":
        asset_boundary_entities += [
            item
            for item in ghost_entities
            if item.name in ("farfield", "symmetric")
            and (param_info.entity_transformation_detected or item.exists(param_info))
        ]
    elif farfield_method in ("quasi-3d", "quasi-3d-periodic"):
        asset_boundary_entities += [
            item for item in ghost_entities if item.name in ("farfield", "symmetric-1", "symmetric-2")
        ]
    elif farfield_method == "user-defined":
        if param_info.use_geometry_AI and param_info.is_beta_mesher:
            # Skip adding "symmetric" ghost if user geometry has y=0 surfaces
            user_sym_surfaces = find_user_symmetry_surfaces(
                asset_boundary_entities,
                param_info.global_bounding_box,
                param_info.planar_face_tolerance,
            )
            if len(user_sym_surfaces) == 0:
                asset_boundary_entities += [
                    item
                    for item in ghost_entities
                    if item.name == "symmetric"
                    and (param_info.entity_transformation_detected or item.exists(param_info))
                ]
    elif farfield_method == "wind-tunnel":
        if param_info.will_generate_forced_symmetry_plane():
            asset_boundary_entities += [item for item in ghost_entities if item.name == "symmetric"]

        wind_tunnel = next(z for z in params.meshing.volume_zones if isinstance(z, WindTunnelFarfield))
        asset_boundary_entities += WindTunnelFarfield._get_valid_ghost_surfaces(
            wind_tunnel.floor_type.type_name,
            wind_tunnel.domain_type,
        )

    return asset_boundary_entities, has_missing_private_attributes


def _collect_zone_zone_interfaces(*, param_info: ParamsValidationInfo, volume_zones: list) -> tuple[set, bool]:
    """Collect potential zone-zone interfaces, and whether any zone comes from a seed point.

    `has_seedpoint_zone` is farfield-agnostic: a `SeedpointVolume`'s extent is only known once the
    mesher flood-fills the seed, so under no farfield method can this predict which boundaries
    survive, and `CustomVolume.ensure_beta_mesher_and_compatible_farfield` admits `auto` and
    `wind-tunnel` alongside `user-defined`. It is not specific to snappyHexMesh either -- the
    GeometryAI / beta mesher workflow builds seed-point zones too.

    The interface set stays gated on `user-defined`, where treating every bounding entity as a
    possible zone-zone interface is the only handle available. Under `auto` / `wind-tunnel` the
    precise set is instead the dual-belonging one -- `enclosed_entities` intersected with the
    bounding entities -- which the caller unions in separately. Blanket-exempting every bounding
    entity there would drop the hard error for an exterior boundary that is merely missing its
    boundary condition.
    """
    has_seedpoint_zone = False
    potential_zone_zone_interfaces: set[str] = set()
    collect_interfaces = param_info.farfield_method == "user-defined"

    for zones in volume_zones:
        # Support new CustomZones container
        if not isinstance(zones, CustomZones):
            continue
        for custom_volume in zones.entities.stored_entities:
            if isinstance(custom_volume, SeedpointVolume):
                has_seedpoint_zone = True
            if collect_interfaces and isinstance(custom_volume, CustomVolume):
                expanded = param_info.expand_entity_list(custom_volume.bounding_entities)
                for boundary in expanded:
                    potential_zone_zone_interfaces.add(boundary.name)

    return potential_zone_zone_interfaces, has_seedpoint_zone


def _collect_unmeshed_boundary_names(*, params, param_info: ParamsValidationInfo, volume_zones: list) -> set[str]:
    """Asset boundaries that no volume zone claims, so the mesher will not carry them over.

    `CustomZones` switches the volume mesher from "mesh every input patch" to an explicit
    whitelist: it pushes the whole input surface mesh into the synthetic farfield zone only
    while no custom zone exists, and otherwise builds each zone from exactly the patches that
    zone lists. A patch no zone lists is simply absent from the volume mesh, so demanding a
    boundary condition for it would reject a legitimate submission.

    A patch is claimed by being a `CustomVolume`'s `bounding_entities`, or by being enclosed by
    any zone that can enclose surfaces -- the farfield, and equally a rotation volume, whose
    `enclosed_entities` the translator emits as `enclosedObjects` so that a blade inside a
    rotating zone is meshed like any other patch. Matched on the field rather than on a list of
    zone classes, so a new enclosing zone type is covered without touching this.

    Only real asset boundaries are considered. Ghost surfaces are not in
    `AssetCache.boundaries`, so forgetting a boundary condition on the farfield ghost stays
    the hard error it is today.

    Empty when no `CustomVolume` bounds a zone. A `SeedpointVolume` contributes nothing here --
    its extent is only known once the mesher flood-fills the seed -- and the orphan claim is
    suppressed outright when one is present, since a boundary cannot be declared unmeshed on
    evidence this side of meshing does not have.
    """
    meshed_surface_ids: set[str] = set()
    for custom_volume_info in param_info.to_be_generated_custom_volumes.values():
        meshed_surface_ids |= custom_volume_info.get("boundary_surface_ids", set())

    if not meshed_surface_ids:
        return set()

    for zone in volume_zones:
        enclosed_entities = getattr(zone, "enclosed_entities", None)
        if enclosed_entities is None:
            continue
        for entity in param_info.expand_entity_list(enclosed_entities):
            meshed_surface_ids.add(entity.private_attribute_id)
            # A `CustomVolume` may itself be enclosed; the translator unwraps it into its own
            # bounding entities, which are then meshed too.
            bounding_entities = getattr(entity, "bounding_entities", None)
            if bounding_entities is not None:
                meshed_surface_ids |= {
                    child.private_attribute_id for child in param_info.expand_entity_list(bounding_entities)
                }

    return {
        boundary.name
        for boundary in (params.private_attribute_asset_cache.boundaries or [])
        if boundary.private_attribute_id not in meshed_surface_ids
    }


def _collect_volume_mesh_interface_names(params) -> set[str]:
    """Collect names of the auto-detected zone-zone interfaces of a volume mesh.

    `VolumeMeshEntityInfo.get_boundaries` (and therefore `AssetCache.boundaries`) omits
    interfaces so that they are never required to carry a boundary condition. They are
    still known `Surface` entities though, and may be assigned a boundary condition to
    replace the interface coupling, so they must be recognized separately.
    """
    entity_info = params.private_attribute_asset_cache.project_entity_info
    if not isinstance(entity_info, VolumeMeshEntityInfo):
        return set()
    return {item.name for item in entity_info.boundaries if item.private_attribute_is_interface}


def _collect_farfield_custom_volume_interfaces(*, param_info: ParamsValidationInfo) -> set[str]:
    """Collect interface names for dual-belonging faces (farfield enclosed_entities ∩ CustomVolume bounding_entities).

    Returns names (not IDs) since _validate_boundary_completeness works with name sets.
    """
    return {param_info.farfield_enclosed_entities[sid] for sid in param_info.farfield_cv_dual_belonging_ids}


def _iter_surface_model_entities(params, param_info: ParamsValidationInfo):
    """Yield every (model, entity) pair referenced by the surface models of `params`."""
    for model in params.models:
        if not isinstance(model, get_args(SurfaceModelTypes)):
            continue

        if hasattr(model, "entities") and model.entities is not None:
            entities = param_info.expand_entity_list(model.entities)
        elif hasattr(model, "entity_pairs") and model.entity_pairs is not None:
            # `Periodic` pair form.
            entities = [pair for surface_pair in model.entity_pairs.items for pair in surface_pair.pair]
        else:
            entities = []

        for entity in entities:
            yield model, entity


def _collect_used_boundary_names(params, param_info: ParamsValidationInfo) -> set:
    """Collect all boundary names referenced in Surface BC models."""
    if len(params.models) == 1 and isinstance(params.models[0], Fluid):
        raise ValueError("No boundary conditions are defined in the `models` section.")

    return {entity.name for _model, entity in _iter_surface_model_entities(params, param_info)}


def _collect_interface_override_names(
    params, param_info: ParamsValidationInfo, interface_boundaries: set[str]
) -> set[str]:
    """Collect zone-zone interfaces whose interface coupling is replaced by a boundary condition.

    `PorousJump` is excluded: it is imposed *on* the coupling (it needs the donor-side state)
    rather than replacing it, and enforces its own pairing requirements.
    """
    return {
        entity.name
        for model, entity in _iter_surface_model_entities(params, param_info)
        if entity.name in interface_boundaries and not isinstance(model, PorousJump)
    }


def _validate_interface_override_zones(params, param_info: ParamsValidationInfo, overridden_interfaces: set) -> None:
    """Only interfaces between two static fluid zones may have their coupling replaced.

    A rotating zone needs its interfaces for the sliding-interface interpolation and a solid
    zone for conjugate heat transfer, so whatever is assigned there is overwritten
    downstream. Only the zone owning each face can be resolved here; the donor side is
    checked in the pipeline, where the mesh metadata provides the interface pairing.
    """
    if not overridden_interfaces:
        return

    entity_info = params.private_attribute_asset_cache.project_entity_info
    zone_of_boundary = {
        boundary_name: zone.name
        for zone in entity_info.zones
        for boundary_name in zone.private_attribute_zone_boundary_names.items
    }

    for model_type, zone_description in ((Rotation, "rotating"), (Solid, "solid")):
        assigned_zone_names = {
            entity.name
            for model in params.models
            if isinstance(model, model_type)
            for entity in param_info.expand_entity_list(model.entities)
        }
        for interface_name in sorted(overridden_interfaces):
            zone_name = zone_of_boundary.get(interface_name)
            if zone_name in assigned_zone_names:
                raise ValueError(
                    f"Boundary `{interface_name}` is a zone-zone interface of {zone_description} zone "
                    f"`{zone_name}`; boundary conditions cannot be assigned to it."
                )


def _has_models_implying_potential_overlap(params, param_info: ParamsValidationInfo) -> bool:
    """Detect models whose presence implies the input geometry/surface mesh may
    have overlapping faces that the mesher will turn into zone-to-zone
    interfaces.

    The Python client cannot detect such overlap at validation time, so for
    these models we cannot assert that a missing-BC face is genuinely missing
    — it may end up consumed by an auto-generated interface. Used to downgrade
    the missing-BC error to a warning.

    Volume-mesh workflows are excluded: meshing has already finished, so any
    boundary in `VolumeMeshEntityInfo.boundaries` that doesn't have a BC is
    genuinely missing and the original strict error must be preserved.

    Returns True if root is geometry/surface_mesh AND any of:
    - A PorousJump model is present (its surfaces imply overlap).
    - A PorousMedium model is present whose entities include a CustomVolume
      (the CustomVolume bounding faces may overlap with farfield/other faces).
    """
    if param_info.root_asset_type == "volume_mesh":
        return False

    for model in params.models:
        if isinstance(model, PorousJump):
            return True
        if isinstance(model, PorousMedium):
            for entity in model.entities.stored_entities:
                if isinstance(entity, CustomVolume):
                    return True
    return False


def _validate_boundary_completeness(
    *,
    asset_boundaries: set,
    used_boundaries: set,
    potential_zone_zone_interfaces: set,
    interface_boundaries: set,
    overridden_interfaces: set,
    has_seedpoint_zone: bool,
    unmeshed_boundaries: set,
    entity_transformation_detected: bool,
    has_missing_private_attributes: bool = False,
    use_geometry_AI: bool = False,
    has_potential_overlap: bool = False,
) -> None:
    """Validate missing/unknown boundary references with error/warning policy.

    `asset_boundaries` is the set of boundaries that must each carry a boundary condition.
    `interface_boundaries` (zone-zone interfaces of a volume mesh) are known entities that
    may carry one but are not required to, so they take part in the unknown-boundary check
    only.
    """
    missing_boundaries = asset_boundaries - used_boundaries - potential_zone_zone_interfaces
    unknown_boundaries = used_boundaries - asset_boundaries - interface_boundaries

    # Deliberately not intersected with `missing_boundaries`: a patch that drops out of the mesh
    # is worth reporting whether or not it carries a boundary condition. Assigning one to it is if
    # anything the more alarming case, since the user clearly expects it to be meshed.
    zone_orphans = asset_boundaries & unmeshed_boundaries
    if zone_orphans and not has_seedpoint_zone:
        missing_boundaries -= zone_orphans
        orphan_list = ", ".join(sorted(zone_orphans))
        add_validation_warning(
            f"The following boundaries are not part of any volume zone and will therefore not "
            f"appear in the volume mesh: {orphan_list}. If they are meant to be meshed, add them "
            "to the `bounding_entities` of a `CustomVolume` or to the `enclosed_entities` of the "
            "farfield or a rotation volume."
        )

    if missing_boundaries and has_seedpoint_zone:
        # A `SeedpointVolume` zone is whatever the mesher's flood fill reaches from the seed, so
        # neither branch below can be justified: the boundary may be kept (making an error right)
        # or unreached (making it wrong). Say what is known and let it through -- the mesher is
        # the only thing that can settle it.
        add_validation_warning(
            f"The following boundaries do not have a boundary condition: "
            f"{', '.join(sorted(missing_boundaries))}. Zones defined by a `SeedpointVolume` are "
            "resolved by a flood fill during meshing, so whether these end up in the volume mesh "
            "cannot be determined here. Add a boundary condition for any that should be meshed."
        )
        missing_boundaries = set()

    if missing_boundaries:
        missing_list = ", ".join(sorted(missing_boundaries))
        if entity_transformation_detected or has_missing_private_attributes or use_geometry_AI or has_potential_overlap:
            message = (
                f"The following boundaries do not have a boundary condition: {missing_list}. "
                "If these boundaries are valid, please add them to a boundary condition model in the `models` section."
            )
            add_validation_warning(message)
        else:
            message = (
                f"The following boundaries do not have a boundary condition: {missing_list}. "
                "Please add them to a boundary condition model in the `models` section."
            )
            raise ValueError(message)

    if unknown_boundaries:
        unknown_list = ", ".join(sorted(unknown_boundaries))
        raise ValueError(
            f"The following boundaries are not known `Surface` "
            f"entities but appear in the `models` section: {unknown_list}."
        )

    if overridden_interfaces:
        interface_list = ", ".join(sorted(overridden_interfaces))
        add_validation_warning(
            f"The following zone-zone interfaces are assigned boundary conditions: {interface_list}."
        )


def _check_complete_boundary_condition_and_unknown_surface(params, param_info):
    # Step 1: Determine whether this check should run
    current_lvls = get_validation_levels() if get_validation_levels() else []
    if all(level not in current_lvls for level in (ALL, CASE)):
        return params

    # Step 2: Collect asset boundaries
    asset_boundary_entities, has_missing_private_attributes = _collect_asset_boundary_entities(params, param_info)
    if asset_boundary_entities is None or asset_boundary_entities == []:
        raise ValueError("[Internal] Failed to retrieve asset boundaries")

    asset_boundaries = {boundary.name for boundary in asset_boundary_entities}
    mirror_status = getattr(params.private_attribute_asset_cache, "mirror_status", None)
    if mirror_status is not None and getattr(mirror_status, "mirrored_surfaces", None):
        asset_boundaries |= {entity.name for entity in mirror_status.mirrored_surfaces}

    # Step 3: Compute special-case interfaces and used boundaries
    volume_zones = _collect_volume_zones(params)
    potential_zone_zone_interfaces, has_seedpoint_zone = _collect_zone_zone_interfaces(
        param_info=param_info, volume_zones=volume_zones
    )
    potential_zone_zone_interfaces |= _collect_farfield_custom_volume_interfaces(param_info=param_info)
    interface_boundaries = _collect_volume_mesh_interface_names(params)
    used_boundaries = _collect_used_boundary_names(params, param_info)
    overridden_interfaces = _collect_interface_override_names(params, param_info, interface_boundaries)
    # Before the completeness policy, so that an invalid override is rejected outright rather
    # than also being warned about as if it were going to take effect.
    _validate_interface_override_zones(params, param_info, overridden_interfaces)

    # Warn if multiple y=0 surfaces have different BC types
    if param_info.farfield_method == "user-defined":
        sym_surfaces = find_user_symmetry_surfaces(
            asset_boundary_entities,
            param_info.global_bounding_box,
            param_info.planar_face_tolerance,
        )
        if len(sym_surfaces) > 1:
            sym_names = {s.name for s in sym_surfaces}
            bc_types = {
                type(m).__name__
                for m in params.models
                if isinstance(m, get_args(SurfaceModelTypes))
                and hasattr(m, "entities")
                and any(e.name in sym_names for e in param_info.expand_entity_list(m.entities))
            }
            if len(bc_types) > 1:
                add_validation_warning(
                    f"Multiple symmetry plane surfaces have different boundary conditions "
                    f"({', '.join(sorted(bc_types))}). Please check if this is intended."
                )

    # Step 4: Validate set differences with policy
    _validate_boundary_completeness(
        asset_boundaries=asset_boundaries,
        used_boundaries=used_boundaries,
        potential_zone_zone_interfaces=potential_zone_zone_interfaces,
        interface_boundaries=interface_boundaries,
        overridden_interfaces=overridden_interfaces,
        has_seedpoint_zone=has_seedpoint_zone,
        unmeshed_boundaries=_collect_unmeshed_boundary_names(
            params=params, param_info=param_info, volume_zones=volume_zones
        ),
        entity_transformation_detected=param_info.entity_transformation_detected,
        has_missing_private_attributes=has_missing_private_attributes,
        use_geometry_AI=param_info.use_geometry_AI,
        has_potential_overlap=_has_models_implying_potential_overlap(params, param_info),
    )

    return params


def _check_parent_volume_is_rotating(models, param_info: ParamsValidationInfo):
    current_lvls = get_validation_levels() if get_validation_levels() else []
    if all(level not in current_lvls for level in (ALL, CASE)):
        return models

    rotating_zone_names = {
        entity.name
        for model in models
        if isinstance(model, Rotation)
        for entity in (param_info.expand_entity_list(model.entities))
    }

    for model_index, model in enumerate(models):
        if isinstance(model, Rotation) is False:
            continue
        if model.parent_volume is None:
            continue
        if model.parent_volume.name not in rotating_zone_names:
            raise ValueError(
                f"For model #{model_index}, the parent rotating volume ({model.parent_volume.name}) is not "
                "used in any other `Rotation` model's `volumes`."
            )
    return models


def _check_and_add_noninertial_reference_frame_flag(params):
    current_lvls = get_validation_levels() if get_validation_levels() else []
    if all(level not in current_lvls for level in (ALL, CASE)):
        return params

    noninertial_reference_frame_default_value = True
    is_steady = True
    if isinstance(params.time_stepping, Unsteady):
        noninertial_reference_frame_default_value = False
        is_steady = False

    models = params.models

    for model_index, model in enumerate(models):
        if isinstance(model, Rotation) is False:
            continue

        if model.rotating_reference_frame_model is None:
            model.rotating_reference_frame_model = noninertial_reference_frame_default_value

        if model.rotating_reference_frame_model is False and is_steady is True:
            raise ValueError(
                f"For model #{model_index}, the rotating_reference_frame_model may not be set to False "
                "for steady state simulations."
            )

    return params


def _collect_rotation_zone_entity_names(params, param_info: ParamsValidationInfo) -> set[str]:
    """Collect entity names registered under RotationVolume/RotationCylinder/RotationSphere."""
    names: set[str] = set()
    for zone in _collect_volume_zones(params):
        if isinstance(zone, (RotationVolume, RotationCylinder, RotationSphere)):
            for entity in param_info.expand_entity_list(zone.entities):
                names.add(entity.name)
    return names


def _workflow_requires_volume_zone_generation(params) -> bool:
    """True when project starts from geometry or surface mesh (mesher must generate volume zones)."""
    cache = params.private_attribute_asset_cache
    if cache is None:
        return False
    return isinstance(cache.project_entity_info, (GeometryEntityInfo, SurfaceMeshEntityInfo))


def _check_rotation_entities_have_volume_zone(params, param_info: ParamsValidationInfo):
    """For geometry/surface-mesh workflows, every Rotation entity that is a generatable shape
    (Cylinder/AxisymmetricBody/Sphere) must have a matching entry under
    `meshing.volume_zones` as RotationVolume/RotationCylinder/RotationSphere so the mesher
    can produce the corresponding volume zone."""
    if not _workflow_requires_volume_zone_generation(params):
        return params
    if not params.models:
        return params

    registered = _collect_rotation_zone_entity_names(params, param_info)

    for model_index, model in enumerate(params.models):
        if not isinstance(model, Rotation):
            continue
        for entity in param_info.expand_entity_list(model.entities):
            if not isinstance(entity, (Cylinder, AxisymmetricBody, Sphere)):
                continue
            if entity.name in registered:
                continue
            raise ValueError(
                f'Rotation model #{model_index} references entity "{entity.name}" of type '
                f"{type(entity).__name__}, but the simulation does not start from a volume mesh "
                "and no matching RotationVolume/RotationCylinder/RotationSphere exists under "
                "`meshing.volume_zones`. Add the same entity to `meshing.volume_zones` so the "
                "volume zone can be generated by the mesher."
            )

    return params


def _check_steady_output_write_count(params):
    if not isinstance(params.time_stepping, Steady) or params.outputs is None:
        return params
    write_count_warning_threshold = 50
    max_steps = params.time_stepping.max_steps
    for output in params.outputs:
        frequency = getattr(output, "frequency", None)
        frequency_offset = getattr(output, "frequency_offset", 0)
        if not frequency or frequency <= 0:
            continue
        approximate_write_count = max(0, max_steps - frequency_offset) // frequency
        if approximate_write_count > write_count_warning_threshold:
            output_name = getattr(output, "name", None) or output.output_type
            add_validation_warning(
                f"`{output_name}` is set to be saved approximately {approximate_write_count} times "
                f"(every {frequency} pseudo steps of {max_steps}) during this steady simulation. "
                "This may generate a large number of files and impact performance; consider "
                "increasing the output frequency."
            )
    return params


def _check_time_average_output(params):
    if isinstance(params.time_stepping, Unsteady) or params.outputs is None:
        return params
    time_average_output_types = set()
    for output in params.outputs:
        if isinstance(output, TimeAverageOutputTypes):
            time_average_output_types.add(output.output_type)
    if len(time_average_output_types) > 0:
        output_type_list = ",".join(f"`{output_type}`" for output_type in sorted(time_average_output_types))
        output_type_list.strip(",")
        raise ValueError(f"{output_type_list} can only be used in unsteady simulations.")
    return params


def _check_valid_models_for_liquid(models, param_info):
    if not models:
        return models
    if param_info.using_liquid_as_material is False:
        return models
    for model in models:
        if isinstance(model, (Inflow, Outflow, Solid)):
            raise ValueError(f"`{model.type}` type model cannot be used when using liquid as simulation material.")
    return models


def _check_duplicate_isosurface_names(outputs):
    if outputs is None:
        return outputs
    isosurface_names = []
    isosurface_time_avg_names = []
    for output in outputs:
        if isinstance(output, IsosurfaceOutput):
            for entity in output.entities.items:
                if entity.name == "qcriterion":
                    raise ValueError(
                        "The name `qcriterion` is reserved for the autovis isosurface from solver, "
                        "please rename the isosurface."
                    )
        if is_exact_instance(output, IsosurfaceOutput):
            for entity in output.entities.items:
                if entity.name in isosurface_names:
                    raise ValueError(
                        f"Another isosurface with name: `{entity.name}` already exists, please rename the isosurface."
                    )
                isosurface_names.append(entity.name)
        if is_exact_instance(output, TimeAverageIsosurfaceOutput):
            for entity in output.entities.items:
                if entity.name in isosurface_time_avg_names:
                    raise ValueError(
                        "Another time average isosurface with name: "
                        f"`{entity.name}` already exists, please rename the isosurface."
                    )
                isosurface_time_avg_names.append(entity.name)
    return outputs


def _check_surface_output_naming(outputs, param_info: ParamsValidationInfo):
    """Validate ``SurfaceOutput`` / ``TimeAverageSurfaceOutput`` naming.

    Rule 1 (shared-surface uniqueness): when the same surface appears in multiple
    instances of the same type, each sharing instance must carry a unique custom name.

    Rule 2 (write-single-file suffix uniqueness): instances of the same type with
    ``write_single_file=True`` must not resolve to the same filename suffix
    (``""`` for default names, ``"_<name>"`` otherwise).
    """
    if outputs is None:
        return outputs

    def _check_shared_surface_uniqueness(outputs, output_type: type[SurfaceOutput] | type[TimeAverageSurfaceOutput]):
        surface_to_outputs: dict[str, list[SurfaceOutput]] = {}
        for output in outputs:
            if not is_exact_instance(output, output_type):
                continue
            # Dedupe per output: duplicate or overlapping entity-list entries must
            # not make a single output look like multiple outputs to Rule 1.
            for entity_name in {e.name for e in param_info.expand_entity_list(output.entities)}:
                surface_to_outputs.setdefault(entity_name, []).append(output)

        for surface_name, shared_outputs in surface_to_outputs.items():
            if len(shared_outputs) <= 1:
                continue
            for output in shared_outputs:
                if output._has_default_name():
                    raise ValueError(
                        f"The surface `{surface_name}` is used in multiple `{output_type.__name__}`s. "
                        "Each output instance that shares the same surface must specify an explicit, "
                        "unique `name`; the default name cannot be used for a shared surface."
                    )
            names = [o.name for o in shared_outputs]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"The surface `{surface_name}` is used in multiple `{output_type.__name__}`s "
                    "that have the same name. Please specify unique `name` values for each "
                    "output instance that shares the same surface."
                )

    def _check_write_single_file_suffix_uniqueness(
        outputs, output_type: type[SurfaceOutput] | type[TimeAverageSurfaceOutput]
    ):
        suffix_to_outputs: dict[str, list] = {}
        for output in outputs:
            if not (is_exact_instance(output, output_type) and output.write_single_file):
                continue
            suffix = "" if output._has_default_name() else f"_{output.name}"
            suffix_to_outputs.setdefault(suffix, []).append(output)

        colliding = sorted(s for s, outs in suffix_to_outputs.items() if len(outs) > 1)
        if colliding:
            raise ValueError(
                f"Multiple `{output_type.__name__}` instances with `write_single_file=True` "
                f"resolve to the same output filename suffix(es): {colliding}. "
                "Please assign unique names to each instance."
            )

    _check_shared_surface_uniqueness(outputs, SurfaceOutput)
    _check_shared_surface_uniqueness(outputs, TimeAverageSurfaceOutput)
    _check_write_single_file_suffix_uniqueness(outputs, SurfaceOutput)
    _check_write_single_file_suffix_uniqueness(outputs, TimeAverageSurfaceOutput)

    return outputs


def _check_duplicate_actuator_disk_cylinder_names(models, param_info: ParamsValidationInfo):
    if not models:
        return models

    def _check_actuator_disk_names(models):
        actuator_disk_names = set()
        for model in models:
            if not isinstance(model, ActuatorDisk):
                continue

            for entity_index, entity in enumerate(param_info.expand_entity_list(model.entities)):
                if entity.name in actuator_disk_names:
                    raise ValueError(
                        f"The ActuatorDisk cylinder name `{entity.name}` at index {entity_index}"
                        f" in model `{model.name}` has already been used."
                        " Please use unique Cylinder entity names among all ActuatorDisk instances."
                    )
                actuator_disk_names.add(entity.name)

    _check_actuator_disk_names(models)

    return models


def _check_unique_selector_names(params):
    """Check that all EntitySelector names are unique across the entire SimulationParams.

    This validator checks the asset_cache.used_selectors field, which is populated
    during the tokenization process in set_up_params_for_uploading().
    """
    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    if asset_cache is None:
        return params

    used_selectors = getattr(asset_cache, "used_selectors", None)
    if not used_selectors:
        return params

    selector_names: set[str] = set()  # name -> first occurrence info

    for selector in used_selectors:
        selector_name = selector.name
        if selector_name in selector_names:
            raise ValueError(f"Duplicate selector name '{selector_name}' found. Each selector must have a unique name.")
        # Store location info for better error messages
        selector_names.add(selector_name)

    return params


def _check_coordinate_system_constraints(params, param_info: ParamsValidationInfo):
    """Validate coordinate system usage constraints.

    1. GeometryBodyGroup assignments require GeometryAI to be enabled.
    2. Entities requiring uniform scaling (Box, Cylinder, AxisymmetricBody)
       must not be assigned to coordinate systems with non-uniform scaling.
    """
    coord_status = params.private_attribute_asset_cache.coordinate_system_status

    # No coordinate systems in use
    if coord_status is None or not coord_status.assignments:
        return params

    # Entity types requiring uniform scaling
    uniform_scale_required_types = {"Box", "Cylinder", "AxisymmetricBody"}

    # Check 1: GAI requirement only for GeometryBodyGroup
    has_geometry_body_group_assignment = False
    for assignment_group in coord_status.assignments:
        for entity_ref in assignment_group.entities:
            if entity_ref.entity_type == "GeometryBodyGroup":
                has_geometry_body_group_assignment = True
                break
        if has_geometry_body_group_assignment:
            break

    if has_geometry_body_group_assignment and not param_info.use_geometry_AI:
        raise ValueError(
            "Coordinate system assignment to GeometryBodyGroup is only supported when Geometry AI is enabled."
        )

    # Check 2: Early validation of uniform scaling for entities that require it
    coordinate_system_state = CoordinateSystemState._from_status(status=coord_status)

    for assignment_group in coord_status.assignments:
        # Get entities that require uniform scaling in this assignment
        entities_requiring_uniform = [
            entity_ref
            for entity_ref in assignment_group.entities
            if entity_ref.entity_type in uniform_scale_required_types
        ]

        if not entities_requiring_uniform:
            continue

        # Get the coordinate system and its composed matrix
        coord_sys = coordinate_system_state._get_coordinate_system_by_id(assignment_group.coordinate_system_id)
        if coord_sys is None:
            continue  # Should not happen if status is valid

        matrix = coordinate_system_state._get_coordinate_system_matrix(coordinate_system=coord_sys)

        if not _is_uniform_scale(matrix):
            scale_factors = _extract_scale_from_matrix(matrix)
            entity_names = [f"{e.entity_type}:{e.entity_id}" for e in entities_requiring_uniform]
            raise ValueError(
                f"Coordinate system '{coord_sys.name}' has non-uniform scaling "
                f"{scale_factors.tolist()}, which is incompatible with entities: "
                f"{entity_names}. Box, Cylinder, and AxisymmetricBody only support "
                f"uniform scaling."
            )

    return params


def _is_constant_gamma_coefficients(coefficients):
    """
    Check if NASA 9-coefficient set represents constant gamma (calorically perfect gas).

    For constant gamma with CompressibleIsentropic solver, only a2 (index 2) should be non-zero.
    All other coefficients (a0, a1, a3-a6, a7, a8) must be zero.

    cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4

    For constant cp compatible with the 4x4 isentropic solver, only a2 should be non-zero.
    """
    tolerance = 1e-10
    # Check all coefficients except a2 (index 2) are zero
    for i in range(9):
        if i == 2:
            continue  # Skip a2, which should be non-zero
        if abs(coefficients[i]) > tolerance:
            return False
    return True


def _has_temperature_dependent_coefficients(temperature_ranges):
    """Check if any temperature range has non-constant-gamma coefficients."""
    for coeff_set in temperature_ranges:
        if not _is_constant_gamma_coefficients(coeff_set.coefficients):
            return True
    return False


def _uses_compressible_isentropic_solver(params):
    """Check if CompressibleIsentropic solver is being used."""
    if not params.models:
        return False
    for model in params.models:
        if isinstance(model, Fluid) and model.navier_stokes_solver.type_name == "CompressibleIsentropic":
            return True
    return False


def _get_gas_material(params):
    """Get Air or Gas material from operating condition, or None if not applicable."""
    if params.operating_condition is None:
        return None
    op = params.operating_condition
    if not hasattr(op, "thermal_state") or op.thermal_state is None:
        return None
    material = op.thermal_state.material
    if isinstance(material, (Air, Gas)):
        return material
    return None


def _material_has_temperature_dependent_gas(material):
    """Check if a Gas material uses a temperature-dependent NASA-9 polynomial.

    Air is CPG by definition so it always returns False. Only checks the
    `thermally_perfect_gas` slot -- the species_transport_model case is handled
    separately by `_check_species_transport_not_with_isentropic_solver`.
    """
    if isinstance(material, Air):
        return False
    tpg = getattr(material, "thermally_perfect_gas", None)
    if tpg is None:
        return False
    for species in tpg.species:
        if _has_temperature_dependent_coefficients(species.nasa_9_coefficients.temperature_ranges):
            return True
    return False


def _check_krylov_solver_restrictions(params):
    """Validate that the Krylov solver is not used with incompatible settings."""
    models = params.models
    if not models:
        return params

    for model in models:
        if not isinstance(model, Fluid):
            continue
        ns = model.navier_stokes_solver
        if not isinstance(ns.linear_solver, KrylovLinearSolver):
            continue

        if ns.limit_velocity:
            raise ValueError(
                "KrylovLinearSolver is not compatible with limit_velocity=True. "
                "Please disable the velocity limiter when using the Krylov solver."
            )
        if ns.limit_pressure_density:
            raise ValueError(
                "KrylovLinearSolver is not compatible with limit_pressure_density=True. "
                "Please disable the pressure-density limiter when using the Krylov solver."
            )
        if params.time_stepping is not None and isinstance(params.time_stepping, Unsteady):
            raise ValueError(
                "KrylovLinearSolver is not supported with Unsteady time stepping. Please use Steady time stepping."
            )

    return params


def _check_tpg_not_with_isentropic_solver(params):
    """
    Validate that temperature-dependent ThermallyPerfectGas is not used with CompressibleIsentropic solver.

    The CompressibleIsentropic solver (4x4 system) does not support true thermally perfect gas
    models where cp varies with temperature. However, it does support constant gamma (CPG)
    coefficients where only the a2 term is non-zero.

    Users must use the full Compressible solver when using temperature-dependent gas properties.
    """
    if not _uses_compressible_isentropic_solver(params):
        return params

    material = _get_gas_material(params)
    if material is None:
        return params

    if _material_has_temperature_dependent_gas(material):
        raise ValueError(
            "Temperature-dependent ThermallyPerfectGas model is not supported with the "
            "CompressibleIsentropic solver. The CompressibleIsentropic solver uses a 4x4 system "
            "that decouples the energy equation and requires constant gamma. "
            "Only constant-gamma coefficients (where only a2 is non-zero) are allowed. "
            "Please use type_name='Compressible' in NavierStokesSolver for thermally perfect gas simulations."
        )

    return params


def _check_species_transport_not_with_isentropic_solver(params):
    """
    Validate that a multi-species transport model is not used with CompressibleIsentropic.

    Variable composition makes R_mix(Y) and cp(Y) spatially varying, so the equation of
    state ``p = rho * R_mix(Y) * T`` has three coupled unknowns rather than two. Closing
    that system requires the energy equation to pin T. The CompressibleIsentropic 4x4
    solver (rho + 3 momentum, no rho*E) decouples energy and assumes constant gamma /
    constant R, so it has no degree of freedom to absorb the variable-composition
    coupling regardless of Mach number.
    """
    if not _uses_compressible_isentropic_solver(params):
        return params

    material = _get_gas_material(params)
    if material is None:
        return params

    if getattr(material, "species_transport_model", None) is not None:
        raise ValueError(
            "A SpeciesTransportModel (variable-composition multi-species transport) is not "
            "supported with the CompressibleIsentropic solver, which uses a 4x4 system that "
            "decouples the energy equation and assumes a single fixed composition. "
            "Please use type_name='Compressible' in NavierStokesSolver for multi-species cases."
        )

    return params
