"""
validation for SimulationParams
"""

from typing import Type, Union, get_args

from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemManager,
)
from flow360.component.simulation.entity_operation import (
    _extract_scale_from_matrix,
    _is_uniform_scale,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
)
from flow360.component.simulation.meshing_param.volume_params import (
    CustomZones,
    WindTunnelFarfield,
)
from flow360.component.simulation.models.solver_numerics import NoneSolver
from flow360.component.simulation.models.surface_models import (
    Inflow,
    Outflow,
    PorousJump,
    SurfaceModelTypes,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    Fluid,
    Rotation,
    Solid,
)
from flow360.component.simulation.outputs.outputs import (
    IsosurfaceOutput,
    ProbeOutput,
    SliceOutput,
    SurfaceOutput,
    TimeAverageIsosurfaceOutput,
    TimeAverageOutputTypes,
    TimeAverageSurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import CustomVolume, SeedpointVolume
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.utils import is_exact_instance
from flow360.component.simulation.validation.validation_context import (
    ALL,
    CASE,
    ParamsValidationInfo,
    add_validation_warning,
    get_validation_levels,
)
from flow360.component.simulation.validation.validation_utils import EntityUsageMap


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
            if isinstance(model, Wall) and model.use_wall_function:
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
            # seen_entity_hashes: set[str] = set()
            for entity in expanded_entities:
                # # pylint: disable=protected-access
                # entity_hash = entity._get_hash()
                # if entity_hash in seen_entity_hashes:
                #     continue
                # if entity_hash is not None:
                #     seen_entity_hashes.add(entity_hash)
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
                preconditioner = model.navier_stokes_solver.low_mach_preconditioner
                if preconditioner:
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
                "You can enable it via model.navier_stokes_solver.low_mach_preconditioner = True for a Fluid "
                "model in the models field of the simulation object."
            )

    return v


def _check_numerical_dissipation_factor_output(v):
    models = v.models

    if models:
        low_dissipation_enabled = False
        for model in models:
            if isinstance(model, Fluid) and model.navier_stokes_solver:
                numerical_dissipation_factor = (
                    model.navier_stokes_solver.numerical_dissipation_factor
                )
                low_dissipation_flag = int(round(1.0 / numerical_dissipation_factor)) - 1
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
                "You can enable it via model.navier_stokes_solver.numerical_dissipation_factor = True for a Fluid "
                "model in the models field of the simulation object."
            )

    return v


def _check_consistency_hybrid_model_volume_output(v):
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

    for output in outputs:
        if isinstance(output, VolumeOutput) and output.output_fields is not None:
            output_fields = output.output_fields.items
            if "SpalartAllmaras_hybridModel" in output_fields and not (
                model_type == "SpalartAllmaras" and run_hybrid_model
            ):
                raise ValueError(
                    "SpalartAllmaras_hybridModel output can only be specified with "
                    "SpalartAllmaras turbulence model and hybrid RANS-LES used."
                )
            if "kOmegaSST_hybridModel" in output_fields and not (
                model_type == "kOmegaSST" and run_hybrid_model
            ):
                raise ValueError(
                    "kOmegaSST_hybridModel output can only be specified with kOmegaSST turbulence model "
                    "and hybrid RANS-LES used."
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
                    if (
                        control.enforcement is not None
                        and turbulence_model_solver.hybrid_model is None
                    ):
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
                output, (SurfaceOutput, VolumeOutput, SliceOutput, ProbeOutput, IsosurfaceOutput)
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
                if (
                    model_solid.material.specific_heat_capacity is None
                    or model_solid.material.density is None
                ):
                    raise ValueError(
                        "In `Solid` model -> material, both `specific_heat_capacity` and `density` "
                        "need to be specified for unsteady simulations."
                    )
                if model_solid.initial_condition is None:
                    raise ValueError(
                        "In `Solid` model, the initial condition needs to be specified "
                        "for unsteady simulations."
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
    # AND the farfield method is one that performs automatic surface deletion (auto/quasi-3d modes)
    # For user-defined/wind-tunnel, missing BCs are always errors since no auto-deletion occurs
    if (
        asset_boundary_entities
        and farfield_method in ("auto", "quasi-3d", "quasi-3d-periodic")
        and all(
            getattr(item, "private_attributes", None) is None for item in asset_boundary_entities
        )
    ):
        has_missing_private_attributes = True

    # Filter out the ones that will be deleted by mesher (only when reliable)
    if not param_info.entity_transformation_detected and not has_missing_private_attributes:
        # pylint:disable=protected-access,duplicate-code
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

    ghost_entities = getattr(
        params.private_attribute_asset_cache.project_entity_info, "ghost_entities", []
    )

    if farfield_method == "auto":
        asset_boundary_entities += [
            item
            for item in ghost_entities
            if item.name in ("farfield", "symmetric")
            and (param_info.entity_transformation_detected or item.exists(param_info))
        ]
    elif farfield_method in ("quasi-3d", "quasi-3d-periodic"):
        asset_boundary_entities += [
            item
            for item in ghost_entities
            if item.name in ("farfield", "symmetric-1", "symmetric-2")
        ]
    elif farfield_method in ("user-defined", "wind-tunnel"):
        if param_info.will_generate_forced_symmetry_plane():
            asset_boundary_entities += [item for item in ghost_entities if item.name == "symmetric"]
        if farfield_method == "wind-tunnel":
            # pylint: disable=protected-access
            asset_boundary_entities += WindTunnelFarfield._get_valid_ghost_surfaces(
                params.meshing.volume_zones[0].floor_type.type_name,
                params.meshing.volume_zones[0].domain_type,
            )

    return asset_boundary_entities, has_missing_private_attributes


def _collect_zone_zone_interfaces(
    *, param_info: ParamsValidationInfo, volume_zones: list
) -> tuple[set, bool]:
    """Collect potential zone-zone interfaces and snappy multizone flag."""
    snappy_multizone = False
    potential_zone_zone_interfaces: set[str] = set()

    if param_info.farfield_method != "user-defined":
        return potential_zone_zone_interfaces, snappy_multizone

    for zones in volume_zones:
        # Support new CustomZones container
        if not isinstance(zones, CustomZones):
            continue
        for custom_volume in zones.entities.stored_entities:
            if isinstance(custom_volume, CustomVolume):
                expanded = param_info.expand_entity_list(custom_volume.boundaries)
                for boundary in expanded:
                    potential_zone_zone_interfaces.add(boundary.name)
            if isinstance(custom_volume, SeedpointVolume):
                # Disable missing boundaries with snappy multizone
                snappy_multizone = True

    return potential_zone_zone_interfaces, snappy_multizone


def _collect_used_boundary_names(params, param_info: ParamsValidationInfo) -> set:
    """Collect all boundary names referenced in Surface BC models."""
    if len(params.models) == 1 and isinstance(params.models[0], Fluid):
        raise ValueError("No boundary conditions are defined in the `models` section.")

    used_boundaries: set[str] = set()

    for model in params.models:
        if not isinstance(model, get_args(SurfaceModelTypes)):
            continue
        if isinstance(model, PorousJump):
            continue

        # pylint: disable=protected-access
        if hasattr(model, "entities"):
            entities = param_info.expand_entity_list(model.entities)
        elif hasattr(model, "entity_pairs"):  # Periodic BC
            entities = [
                pair for surface_pair in model.entity_pairs.items for pair in surface_pair.pair
            ]
        else:
            entities = []

        for entity in entities:
            used_boundaries.add(entity.name)

    return used_boundaries


def _validate_boundary_completeness(  # pylint:disable=too-many-arguments
    *,
    asset_boundaries: set,
    used_boundaries: set,
    potential_zone_zone_interfaces: set,
    snappy_multizone: bool,
    entity_transformation_detected: bool,
    has_missing_private_attributes: bool = False,
    use_geometry_AI: bool = False,
) -> None:
    """Validate missing/unknown boundary references with error/warning policy."""
    missing_boundaries = asset_boundaries - used_boundaries - potential_zone_zone_interfaces
    unknown_boundaries = used_boundaries - asset_boundaries

    if missing_boundaries and not snappy_multizone:
        missing_list = ", ".join(sorted(missing_boundaries))
        if entity_transformation_detected or has_missing_private_attributes or use_geometry_AI:
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


def _check_complete_boundary_condition_and_unknown_surface(
    params, param_info
):  # pylint:disable=too-many-branches, too-many-locals,too-many-statements
    # Step 1: Determine whether this check should run
    current_lvls = get_validation_levels() if get_validation_levels() else []
    if all(level not in current_lvls for level in (ALL, CASE)):
        return params

    # Step 2: Collect asset boundaries
    asset_boundary_entities, has_missing_private_attributes = _collect_asset_boundary_entities(
        params, param_info
    )
    if asset_boundary_entities is None or asset_boundary_entities == []:
        raise ValueError("[Internal] Failed to retrieve asset boundaries")

    asset_boundaries = {boundary.name for boundary in asset_boundary_entities}
    mirror_status = getattr(params.private_attribute_asset_cache, "mirror_status", None)
    if mirror_status is not None and getattr(mirror_status, "mirrored_surfaces", None):
        asset_boundaries |= {entity.name for entity in mirror_status.mirrored_surfaces}

    # Step 3: Compute special-case interfaces and used boundaries
    volume_zones = _collect_volume_zones(params)
    potential_zone_zone_interfaces, snappy_multizone = _collect_zone_zone_interfaces(
        param_info=param_info, volume_zones=volume_zones
    )
    used_boundaries = _collect_used_boundary_names(params, param_info)

    # Step 4: Validate set differences with policy
    _validate_boundary_completeness(
        asset_boundaries=asset_boundaries,
        used_boundaries=used_boundaries,
        potential_zone_zone_interfaces=potential_zone_zone_interfaces,
        snappy_multizone=snappy_multizone,
        entity_transformation_detected=param_info.entity_transformation_detected,
        has_missing_private_attributes=has_missing_private_attributes,
        use_geometry_AI=param_info.use_geometry_AI,
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


def _check_time_average_output(params):
    if isinstance(params.time_stepping, Unsteady) or params.outputs is None:
        return params
    time_average_output_types = set()
    for output in params.outputs:
        if isinstance(output, TimeAverageOutputTypes):
            time_average_output_types.add(output.output_type)
    if len(time_average_output_types) > 0:
        output_type_list = ",".join(
            f"`{output_type}`" for output_type in sorted(time_average_output_types)
        )
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
            raise ValueError(
                f"`{model.type}` type model cannot be used when using liquid as simulation material."
            )
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


def _check_duplicate_surface_usage(outputs, param_info: ParamsValidationInfo):
    if outputs is None:
        return outputs

    def _check_surface_usage(
        outputs, output_type: Union[Type[SurfaceOutput], Type[TimeAverageSurfaceOutput]]
    ):
        surface_names = set()
        for output in outputs:
            if not is_exact_instance(output, output_type):
                continue
            for entity in param_info.expand_entity_list(output.entities):
                if entity.name in surface_names:
                    raise ValueError(
                        f"The same surface `{entity.name}` is used in multiple `{output_type.__name__}`s."
                        " Please specify all settings for the same surface in one output."
                    )
                surface_names.add(entity.name)

    _check_surface_usage(outputs, SurfaceOutput)
    _check_surface_usage(outputs, TimeAverageSurfaceOutput)

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
            raise ValueError(
                f"Duplicate selector name '{selector_name}' found. "
                f"Each selector must have a unique name."
            )
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
            "Coordinate system assignment to GeometryBodyGroup "
            "is only supported when Geometry AI is enabled."
        )

    # Check 2: Early validation of uniform scaling for entities that require it
    manager = CoordinateSystemManager._from_status(  # pylint: disable=protected-access
        status=coord_status
    )

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
        coord_sys = manager._get_coordinate_system_by_id(  # pylint: disable=protected-access
            assignment_group.coordinate_system_id
        )
        if coord_sys is None:
            continue  # Should not happen if status is valid

        matrix = manager._get_coordinate_system_matrix(  # pylint: disable=protected-access
            coordinate_system=coord_sys
        )

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
