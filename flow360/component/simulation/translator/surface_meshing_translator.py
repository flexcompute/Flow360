"""Surface meshing parameter translator."""

from copy import deepcopy
from typing import List

from unyt import unyt_array

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.meshing_specs import OctreeSpacing
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    UniformRefinement,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    SeedpointVolume,
    SnappyBody,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    ensure_meshing_is_specified,
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
    using_snappy,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.exceptions import Flow360TranslationError
from flow360.log import log


# pylint: disable=invalid-name
def SurfaceEdgeRefinement_to_edges(obj: SurfaceEdgeRefinement):
    """
    Translate SurfaceEdgeRefinement to edges.
    """
    if obj.method.type == "angle":
        return {
            "type": "aniso",
            "method": "angle",
            "value": obj.method.value.to("degree").value.item(),
        }

    if obj.method.type == "height":
        return {
            "type": "aniso",
            "method": "height",
            "value": obj.method.value.value.item(),
        }

    if obj.method.type == "aspectRatio":
        return {"type": "aniso", "method": "aspectRatio", "value": obj.method.value}

    if obj.method.type == "projectAnisoSpacing":
        return {"type": "projectAnisoSpacing"}

    raise Flow360TranslationError(
        error_message=f"Unknown `SurfaceEdgeRefinement` type: {obj.method.type}",
        input_value=obj,
        location=["meshing", "refinements"],
    )


# pylint: disable=invalid-name
def SurfaceRefinement_to_faces(
    obj: SurfaceRefinement, global_max_edge_length, global_curvature_resolution_angle
):
    """
    Translate SurfaceRefinement to faces.
    """
    return {
        "maxEdgeLength": (
            obj.max_edge_length.value.item()
            if obj.max_edge_length is not None
            else global_max_edge_length.value.item()
        ),
        "curvatureResolutionAngle": (
            obj.curvature_resolution_angle.to("degree").value.item()
            if obj.curvature_resolution_angle is not None
            else global_curvature_resolution_angle.to("degree").value.item()
        ),
    }


def remove_numerical_noise_from_spacing(spacing: LengthType, spacing_system: OctreeSpacing):
    """
    If the spacing is in the proximity of 1e-8 to one of the octree series spacing casts that spacing onto the series.
    """
    unit_in = spacing.units
    direct = spacing_system.to_level(spacing)[1]
    if direct:
        return spacing_system[spacing_system.to_level(spacing)[0]].to(unit_in)
    return spacing


def apply_SnappyBodyRefinement(
    refinement: snappy.BodyRefinement, translated, spacing_system: OctreeSpacing
):
    """
    Translate SnappyBodyRefinement to bodies.
    """
    applicable_bodies = [entity.name for entity in refinement.entities.stored_entities]
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_bodies:
            if refinement.gap_resolution is not None:
                body["gap"] = refinement.gap_resolution.value.item()
            if refinement.proximity_spacing is not None:
                body["gapSpacingReduction"] = remove_numerical_noise_from_spacing(
                    refinement.proximity_spacing, spacing_system
                ).value.item()
            if refinement.min_spacing is not None:
                body["spacing"]["min"] = remove_numerical_noise_from_spacing(
                    refinement.min_spacing, spacing_system
                ).value.item()
            if refinement.max_spacing is not None:
                body["spacing"]["max"] = remove_numerical_noise_from_spacing(
                    refinement.max_spacing, spacing_system
                ).value.item()


def get_applicable_regions_dict(refinement_regions):
    """
    Get regions to apply a refinement on.
    """
    applicable_regions = {}
    if refinement_regions:
        for entity in refinement_regions.stored_entities:
            if not isinstance(entity, Surface):
                continue
            split = entity.name.split("::")
            body = split[0]
            if len(split) == 2:
                region = split[1]
            else:
                applicable_regions[body] = None
                continue

            if body in applicable_regions:
                applicable_regions[body].append(region)
            else:
                applicable_regions[body] = [region]

    return applicable_regions


def apply_SnappySurfaceEdgeRefinement(
    refinement: snappy.SurfaceEdgeRefinement,
    translated,
    defaults,
    spacing_system: OctreeSpacing,
):
    """
    Translate SnappySurfaceEdgeRefinement to bodies and regions.
    """
    edges = {"includedAngle": refinement.included_angle.to("degree").value.item()}
    if refinement.min_elem is not None:
        edges["minElem"] = refinement.min_elem
    if refinement.min_len is not None:
        edges["minLen"] = refinement.min_len.value.item()
    if refinement.retain_on_smoothing is not None:
        edges["retainOnSmoothing"] = refinement.retain_on_smoothing
    edges["geometricTestOnly"] = refinement.geometric_test_only
    if refinement.spacing is None:
        edges["edgeSpacing"] = defaults.min_spacing.value.item()
    elif isinstance(refinement.spacing, unyt_array) and isinstance(
        refinement.distances, unyt_array
    ):
        edges["edgeSpacing"] = [
            [
                dist.value.item(),
                remove_numerical_noise_from_spacing(spac, spacing_system).value.item(),
            ]
            for (dist, spac) in zip(refinement.distances, refinement.spacing)
        ]
    else:
        edges["edgeSpacing"] = remove_numerical_noise_from_spacing(
            refinement.spacing, spacing_system
        ).value.item()
    applicable_bodies = (
        [
            entity.name
            for entity in refinement.entities.stored_entities
            if isinstance(entity, SnappyBody)
        ]
        if refinement.entities is not None
        else []
    )
    applicable_regions = get_applicable_regions_dict(refinement_regions=refinement.entities)
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_bodies or (
            body["bodyName"] in applicable_regions and applicable_regions[body["bodyName"]] is None
        ):
            body["edges"] = edges
        if body["bodyName"] in applicable_regions:
            for region in body.get("regions", []):
                if region["patchName"] in applicable_regions[body["bodyName"]]:
                    region["edges"] = edges


def apply_SnappyRegionRefinement(
    refinement: snappy.RegionRefinement, translated, spacing_system: OctreeSpacing
):
    """
    Translate SnappyRegionRefinement to applicable regions.
    """
    applicable_regions = get_applicable_regions_dict(refinement_regions=refinement.entities)
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_regions:
            for region in body.get("regions", []):
                if region["patchName"] in applicable_regions[body["bodyName"]]:
                    if refinement.proximity_spacing is not None:
                        region["gapSpacingReduction"] = remove_numerical_noise_from_spacing(
                            refinement.proximity_spacing, spacing_system
                        ).value.item()

                    region["spacing"] = {
                        "min": remove_numerical_noise_from_spacing(
                            refinement.min_spacing, spacing_system
                        ).value.item(),
                        "max": remove_numerical_noise_from_spacing(
                            refinement.max_spacing, spacing_system
                        ).value.item(),
                    }


def apply_UniformRefinement_w_snappy(
    refinement: UniformRefinement, translated, spacing_system: OctreeSpacing
):
    """
    Translate UniformRefinement to defined volumetric regions.
    """
    if "refinementVolumes" not in translated["geometry"]:
        translated["geometry"]["refinementVolumes"] = []

    for volume in refinement.entities.stored_entities:
        volume_body = {
            "spacing": remove_numerical_noise_from_spacing(
                refinement.spacing, spacing_system
            ).value.item(),
            "name": volume.name,
        }
        if isinstance(volume, Box):
            volume_body["type"] = "box"
            volume_body["min"] = {
                "x": volume.center[0].value.item() - 0.5 * volume.size[0].value.item(),
                "y": volume.center[1].value.item() - 0.5 * volume.size[1].value.item(),
                "z": volume.center[2].value.item() - 0.5 * volume.size[2].value.item(),
            }
            volume_body["max"] = {
                "x": volume.center[0].value.item() + 0.5 * volume.size[0].value.item(),
                "y": volume.center[1].value.item() + 0.5 * volume.size[1].value.item(),
                "z": volume.center[2].value.item() + 0.5 * volume.size[2].value.item(),
            }
        elif isinstance(volume, Cylinder):
            volume_body["type"] = "cylinder"
            volume_body["radius"] = volume.outer_radius.value.item()
            volume_body["point1"] = {
                "x": volume.center[0].value.item()
                - 0.5 * volume.axis[0] * volume.height.value.item(),
                "y": volume.center[1].value.item()
                - 0.5 * volume.axis[1] * volume.height.value.item(),
                "z": volume.center[2].value.item()
                - 0.5 * volume.axis[2] * volume.height.value.item(),
            }

            volume_body["point2"] = {
                "x": volume.center[0].value.item()
                + 0.5 * volume.axis[0] * volume.height.value.item(),
                "y": volume.center[1].value.item()
                + 0.5 * volume.axis[1] * volume.height.value.item(),
                "z": volume.center[2].value.item()
                + 0.5 * volume.axis[2] * volume.height.value.item(),
            }

        else:
            raise Flow360TranslationError(
                f"Volume of type {type(volume)} cannot be used with Snappy.",
                None,
                ["meshing", "surface_meshing"],
            )

        translated["geometry"]["refinementVolumes"].append(volume_body)


def _none_tolerant_min(current, candidate):
    """Return the smaller of two spacing quantities, comparing by raw value."""
    if candidate is not None and candidate < current:
        return candidate
    return current


def _get_effective_min_spacing(input_params, spacing_system: OctreeSpacing):
    """
    Get the effective minimum spacing across all refinements,
    taking proximity_spacing (gap spacing reduction), edge spacings, and
    projected volume refinements into account.
    The result is cast to the nearest lower spacing in the octree series.
    """
    surface_meshing_params = input_params.meshing.surface_meshing
    min_spacing = surface_meshing_params.defaults.min_spacing

    for refinement in surface_meshing_params.refinements or []:
        if isinstance(refinement, (snappy.BodyRefinement, snappy.RegionRefinement)):
            min_spacing = _none_tolerant_min(min_spacing, refinement.min_spacing)
            min_spacing = _none_tolerant_min(min_spacing, refinement.proximity_spacing)
        elif (
            isinstance(refinement, snappy.SurfaceEdgeRefinement) and refinement.spacing is not None
        ):
            edge_spacing = (
                refinement.spacing[0]
                if isinstance(refinement.spacing, unyt_array)
                and isinstance(refinement.distances, unyt_array)
                and len(refinement.spacing) > 0
                else refinement.spacing
            )
            min_spacing = _none_tolerant_min(min_spacing, edge_spacing)
        elif isinstance(refinement, UniformRefinement):
            min_spacing = _none_tolerant_min(min_spacing, refinement.spacing)

    # Also consider projected volume meshing refinements
    if input_params.meshing.volume_meshing is not None:
        for refinement in input_params.meshing.volume_meshing.refinements:
            if isinstance(refinement, UniformRefinement) and refinement.project_to_surface in [
                True,
                None,
            ]:
                min_spacing = _none_tolerant_min(min_spacing, refinement.spacing)

    # Cast to the nearest lower spacing in the octree series
    level = spacing_system.to_level(min_spacing)[0]
    return spacing_system[level].value.item()


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
def snappy_mesher_json(input_params: SimulationParams):
    """
    Get config JSON for snappyHexMesh surface meshing.
    """
    translated = {}
    surface_meshing_params = input_params.meshing.surface_meshing
    # spacing system
    spacing_system: OctreeSpacing = surface_meshing_params.octree_spacing

    # extract geometry information in body: {patch0, ...} format
    bodies = {}
    for face_id in input_params.private_attribute_asset_cache.project_entity_info.all_face_ids:
        solid = face_id.split("::")
        if solid[0] not in bodies:
            bodies[solid[0]] = set()
        if len(solid) == 2:
            bodies[solid[0]].add(solid[1])

    # Fill with defaults
    common_defaults = {
        "gap": surface_meshing_params.defaults.gap_resolution.value.item(),
        "spacing": {
            "min": remove_numerical_noise_from_spacing(
                surface_meshing_params.defaults.min_spacing, spacing_system
            ).value.item(),
            "max": remove_numerical_noise_from_spacing(
                surface_meshing_params.defaults.max_spacing, spacing_system
            ).value.item(),
        },
    }
    translated["geometry"] = {
        "bodies": [
            {
                "bodyName": name,
                **deepcopy(common_defaults),
                "regions": [{"patchName": region} for region in regions],
            }
            for (name, regions) in bodies.items()
        ]
    }

    # sort the lists

    translated["geometry"]["bodies"].sort(key=lambda x: x["bodyName"])

    for body in translated["geometry"]["bodies"]:
        if body["regions"]:
            body["regions"].sort(key=lambda x: x["patchName"])

    # apply refinements
    for refinement in (
        surface_meshing_params.refinements if surface_meshing_params.refinements is not None else []
    ):
        if isinstance(refinement, snappy.BodyRefinement):
            apply_SnappyBodyRefinement(refinement, translated, spacing_system)
        elif isinstance(refinement, snappy.SurfaceEdgeRefinement):
            apply_SnappySurfaceEdgeRefinement(
                refinement, translated, surface_meshing_params.defaults, spacing_system
            )
        elif isinstance(refinement, snappy.RegionRefinement):
            apply_SnappyRegionRefinement(refinement, translated, spacing_system)
        elif isinstance(refinement, UniformRefinement):
            apply_UniformRefinement_w_snappy(refinement, translated, spacing_system)
        else:
            raise Flow360TranslationError(
                f"Refinement of type {type(refinement)} cannot be used with Snappy.",
                None,
                ["meshing", "surface_meshing"],
            )

    # apply projected volumetric refinements
    if input_params.meshing.volume_meshing is not None:
        for refinement in input_params.meshing.volume_meshing.refinements:
            if isinstance(refinement, UniformRefinement) and refinement.project_to_surface in [
                True,
                None,
            ]:
                apply_UniformRefinement_w_snappy(refinement, translated, spacing_system)

    # apply settings
    castellated_mesh_controls = surface_meshing_params.castellated_mesh_controls
    snap_controls = surface_meshing_params.snap_controls
    quality_settings = surface_meshing_params.quality_metrics
    translated["mesherSettings"] = {
        "snappyHexMesh": {
            "castellatedMeshControls": {
                "resolveFeatureAngle": castellated_mesh_controls.resolve_feature_angle.to(
                    "degree"
                ).value.item(),
                "nCellsBetweenLevels": castellated_mesh_controls.n_cells_between_levels,
                "minRefinementCells": castellated_mesh_controls.min_refinement_cells,
            },
            "snapControls": {
                "nSmoothPatch": snap_controls.n_smooth_patch,
                "tolerance": snap_controls.tolerance,
                "nSolveIter": snap_controls.n_solve_iterations,
                "nRelaxIter": snap_controls.n_relax_iterations,
                "nFeatureSnapIter": snap_controls.n_feature_snap_iterations,
                "multiRegionFeatureSnap": snap_controls.multi_region_feature_snap,
                "strictRegionSnap": snap_controls.strict_region_snap,
            },
        },
        "meshQuality": {
            "maxNonOrtho": (
                quality_settings.max_non_orthogonality.to("degree").value.item()
                if quality_settings.max_non_orthogonality
                else 180
            ),
            "maxBoundarySkewness": (
                quality_settings.max_boundary_skewness.to("degree").value.item()
                if quality_settings.max_boundary_skewness
                else -1
            ),
            "maxInternalSkewness": (
                quality_settings.max_internal_skewness.to("degree").value.item()
                if quality_settings.max_internal_skewness
                else -1
            ),
            "maxConcave": (
                quality_settings.max_concavity.to("degree").value.item()
                if quality_settings.max_concavity
                else 180
            ),
            "minVol": (
                -1e30
                if quality_settings.min_pyramid_cell_volume is False
                else (
                    quality_settings.min_pyramid_cell_volume
                    if quality_settings.min_pyramid_cell_volume is not None
                    else (1e-10 * (_get_effective_min_spacing(input_params, spacing_system) ** 3))
                )
            ),
            "minTetQuality": (
                quality_settings.min_tetrahedron_quality
                if quality_settings.min_tetrahedron_quality
                else -1e30
            ),
            "minArea": (
                quality_settings.min_face_area.value.item()
                if quality_settings.min_face_area
                else (1e-12 if quality_settings.min_face_area is None else -1)
            ),
            "minTwist": (quality_settings.min_twist if quality_settings.min_twist else -2),
            "minDeterminant": (
                quality_settings.min_cell_determinant
                if quality_settings.min_cell_determinant
                else -1e5
            ),
            "minVolRatio": (
                quality_settings.min_volume_ratio if quality_settings.min_volume_ratio else 0
            ),
            "minFaceWeight": (
                quality_settings.min_face_weight if quality_settings.min_face_weight else 0
            ),
            "minTriangleTwist": (
                quality_settings.min_triangle_twist if quality_settings.min_triangle_twist else -1
            ),
            "nSmoothScale": (
                quality_settings.n_smooth_scale if quality_settings.n_smooth_scale else 0
            ),
            "errorReduction": (
                quality_settings.error_reduction if quality_settings.error_reduction else 0
            ),
            "zMetricThreshold": (
                0
                if quality_settings.zmetric_threshold is False
                else quality_settings.zmetric_threshold
            ),
            "featureEdgeDeduplicationTolerance": (
                0
                if quality_settings.feature_edge_deduplication_tolerance is False
                else quality_settings.feature_edge_deduplication_tolerance
            ),
            "minVolCollapseRatio": (
                quality_settings.min_volume_collapse_ratio
                if quality_settings.min_volume_collapse_ratio
                else 0
            ),
        },
    }
    # smoothing settings
    smoothing_settings = surface_meshing_params.smooth_controls

    if smoothing_settings:
        translated["smoothingControls"] = {
            "lambda": (smoothing_settings.lambda_factor if smoothing_settings.lambda_factor else 0),
            "mu": (smoothing_settings.mu_factor if smoothing_settings.mu_factor else 0),
            "iter": (smoothing_settings.iterations if smoothing_settings.iterations else 0),
        }
    else:
        translated["smoothingControls"] = {
            "lambda": 0,
            "mu": 0,
            "iter": 0,
        }
    # enforced spacing
    translated["enforcedSpacing"] = spacing_system.base_spacing.value.item()

    # get seedpoint zones
    zones = input_params.meshing.zones

    all_seedpoint_zones = []

    # cad is fluid
    for zone in zones:
        if isinstance(zone, AutomatedFarfield):
            translated["cadIsFluid"] = False

        if isinstance(zone, CustomZones):
            for subzone in zone.entities.stored_entities:
                if isinstance(subzone, SeedpointVolume):
                    all_seedpoint_zones.append(subzone)

    if all_seedpoint_zones:
        translated["cadIsFluid"] = True

    if "cadIsFluid" not in translated:
        raise Flow360TranslationError(
            "Farfield type not specified.", None, ["meshing", "surface_meshing"]
        )

    # points in mesh
    if all_seedpoint_zones and translated["cadIsFluid"]:
        translated["locationInMesh"] = {
            zone.name: [point.value.item() for point in zone.point_in_mesh]
            for zone in all_seedpoint_zones
        }

    return translated


def legacy_mesher_json(input_params: SimulationParams):
    """
    Get JSON for surface meshing.
    """
    translated = {}
    # pylint: disable=duplicate-code
    ##:: >>  Step 1:  Get global maxEdgeLength [REQUIRED]
    if input_params.meshing.defaults.surface_max_edge_length is None:
        log.info("No `surface_max_edge_length` found in the defaults. Skipping translation.")
        raise Flow360TranslationError(
            "No `surface_max_edge_length` found in the defaults",
            input_value=None,
            location=["meshing", "refinements", "defaults"],
        )

    default_max_edge_length = input_params.meshing.defaults.surface_max_edge_length.value.item()

    ##:: >> Step 2: Get curvatureResolutionAngle [REQUIRED]
    translated["curvatureResolutionAngle"] = (
        input_params.meshing.defaults.curvature_resolution_angle.to("degree").value.item()
    )

    ##:: >> Step 3: Get growthRate [REQUIRED]
    translated["growthRate"] = input_params.meshing.defaults.surface_edge_growth_rate

    ##:: >> Step 4: Get edges [OPTIONAL]
    edge_config = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements,
        SurfaceEdgeRefinement,
        translation_func=SurfaceEdgeRefinement_to_edges,
        use_sub_item_as_key=True,
    )
    if edge_config != {}:
        translated["edges"] = edge_config

    ##:: >> Step 5: Get faces
    face_config = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements,
        SurfaceRefinement,
        translation_func=SurfaceRefinement_to_faces,
        translation_func_global_max_edge_length=input_params.meshing.defaults.surface_max_edge_length,
        translation_func_global_curvature_resolution_angle=input_params.meshing.defaults.curvature_resolution_angle,
        use_sub_item_as_key=True,
    )

    ##:: >> Step 5.1: Apply default_max_edge_length to faces that are not explicitly specified
    assert input_params.private_attribute_asset_cache.project_entity_info is not None
    assert isinstance(
        input_params.private_attribute_asset_cache.project_entity_info,
        GeometryEntityInfo,
    )

    for face_id in input_params.private_attribute_asset_cache.project_entity_info.all_face_ids:
        if face_id not in face_config:
            face_config[face_id] = {"maxEdgeLength": default_max_edge_length}

    translated["faces"] = face_config

    ##:: >> Step 6: Tell surface mesher how do we group boundaries.
    translated["boundaries"] = {}
    grouped_faces: List[Surface] = (
        input_params.private_attribute_asset_cache.project_entity_info.get_boundaries()
    )
    for surface in grouped_faces:
        for face_id in surface.private_attribute_sub_components:
            translated["boundaries"][face_id] = {"boundaryName": surface.name}

    return translated


def _get_surface_refinements(refinement_list: list[dict]):
    """
    Get the surface refinements from the input_params.
    """
    return [
        item
        for item in refinement_list
        if item["refinement_type"]
        in ("SurfaceRefinement", "UniformRefinement", "GeometryRefinement")
    ]


def _get_volume_zones(volume_zones_list: list[dict]):
    """
    Get the volume zones from the input_params.
    """
    volume_zones_translated = []
    for item in volume_zones_list:
        if item["type"] in (
            "AutomatedFarfield",
            "UserDefinedFarfield",
            "WindTunnelFarfield",
        ):
            volume_zones_translated.append(item)
        elif item["type"] in (
            "RotationVolume",
            "RotationCylinder",
        ):
            item.pop("stationary_enclosed_entities", None)
            volume_zones_translated.append(item)

    return volume_zones_translated


def _filter_mirror_status(data):
    """Process mirror_status to ensure idempotency while preserving mirroring relationships.

    Strategy:
    - For mirror planes: Replace UUID-based private_attribute_id with deterministic "mirror-{name}"
    - For mirrored entities: Strip their own private_attribute_id, but keep mirror_plane_id
      and update it to match the deterministic format

    This preserves the relationship between mirrored entities and their mirror planes
    while ensuring the translation is idempotent.
    """
    if not isinstance(data, dict):
        return data

    result = {}
    uuid_to_deterministic = {}

    # Process mirror_planes: replace private_attribute_id with deterministic ID
    if "mirror_planes" in data:
        result["mirror_planes"] = []
        for plane in data["mirror_planes"]:
            if not isinstance(plane, dict):
                result["mirror_planes"].append(plane)
                continue

            plane_copy = plane.copy()
            old_id = plane_copy.get("private_attribute_id")
            name = plane_copy.get("name")

            if old_id and name:
                new_id = f"mirror-{name}"
                uuid_to_deterministic[old_id] = new_id
                plane_copy["private_attribute_id"] = new_id

            result["mirror_planes"].append(plane_copy)

    # Process mirrored_geometry_body_groups: remove private_attribute_id, update mirror_plane_id
    if "mirrored_geometry_body_groups" in data:
        result["mirrored_geometry_body_groups"] = []
        for entity in data["mirrored_geometry_body_groups"]:
            if not isinstance(entity, dict):
                result["mirrored_geometry_body_groups"].append(entity)
                continue

            entity_copy = {k: v for k, v in entity.items() if k != "private_attribute_id"}
            if "mirror_plane_id" in entity_copy:
                old_plane_id = entity_copy["mirror_plane_id"]
                entity_copy["mirror_plane_id"] = uuid_to_deterministic.get(
                    old_plane_id, old_plane_id
                )

            result["mirrored_geometry_body_groups"].append(entity_copy)

    # Process mirrored_surfaces: remove private_attribute_id, update mirror_plane_id
    if "mirrored_surfaces" in data:
        result["mirrored_surfaces"] = []
        for entity in data["mirrored_surfaces"]:
            if not isinstance(entity, dict):
                result["mirrored_surfaces"].append(entity)
                continue

            entity_copy = {k: v for k, v in entity.items() if k != "private_attribute_id"}
            if "mirror_plane_id" in entity_copy:
                old_plane_id = entity_copy["mirror_plane_id"]
                entity_copy["mirror_plane_id"] = uuid_to_deterministic.get(
                    old_plane_id, old_plane_id
                )

            result["mirrored_surfaces"].append(entity_copy)

    # Copy over any other keys that might exist
    for k, v in data.items():
        if k not in result:
            result[k] = v

    return result


def _get_gai_setting_whitelist(input_params: SimulationParams) -> dict:
    """
    Generate GAI whitelist with conditional fields based on simulation context.

    Args:
        input_params: The simulation parameters to determine which fields to include.

    Returns:
        A whitelist dictionary for filtering the simulation JSON.
    """
    # Check if rotation zones are present
    has_rotation_zones = False
    if input_params.meshing and input_params.meshing.volume_zones:
        has_rotation_zones = any(
            zone.__class__.__name__ in ("RotationCylinder", "RotationVolume")
            for zone in input_params.meshing.volume_zones
        )

    # Build defaults whitelist
    defaults_whitelist = {
        "surface_max_edge_length": None,
        "curvature_resolution_angle": None,
        "surface_edge_growth_rate": None,
        "geometry_accuracy": None,
        "resolve_face_boundaries": None,
        "preserve_thin_geometry": None,
        "surface_max_aspect_ratio": None,
        "surface_max_adaptation_iterations": None,
        "sealing_size": None,
        "remove_hidden_geometry": None,
        "min_passage_size": None,
        "planar_face_tolerance": None,
    }

    # Conditionally add sliding_interface_tolerance only when rotation zones are present
    if has_rotation_zones:
        defaults_whitelist["sliding_interface_tolerance"] = None

    return {
        "meshing": {
            "defaults": defaults_whitelist,
            "refinements": _get_surface_refinements,
            "volume_zones": _get_volume_zones,
        },
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "face_group_tag": None,
                "face_attribute_names": None,
                "grouped_faces": None,
                "body_group_tag": None,
                "body_attribute_names": None,
                "grouped_bodies": None,
            },
            "mirror_status": _filter_mirror_status,
        },
    }


def _traverse_and_filter(data, whitelist):
    """
    Recursively traverse data and whitelist to extract matching values.

    Args:
        data: The data to traverse
        whitelist: The whitelist structure defining what to extract

    Returns:
        Filtered data matching the whitelist structure
    """
    if isinstance(whitelist, dict):
        result = {}
        for key, value in whitelist.items():
            if key in data:
                if value is None:
                    # Copy as is
                    result[key] = data[key]
                elif callable(value):
                    # Run the function
                    result[key] = value(data[key])
                else:
                    # Recursively traverse
                    result[key] = _traverse_and_filter(data[key], value)
        return result
    return data


def _inject_body_group_transformations_for_mesher(
    *,
    json_data: dict,
    input_params: SimulationParams,
    mesh_unit,  # pylint: disable=unused-argument
) -> None:
    """
    Inject body-group transformation payload expected by the mesher.

    The user-facing `GeometryBodyGroup.transformation` field has been removed. Mesher translation
    still needs per-body-group 3x4 transformation matrices; those are now sourced from coordinate
    system assignments.

    Only the `private_attribute_matrix` field is injected - the mesher only needs the final
    transformation matrix, not the intermediate parameters (origin, axis_of_rotation, etc.).
    """
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.draft_context.coordinate_system_manager import (
        CoordinateSystemManager,
    )

    asset_cache = input_params.private_attribute_asset_cache
    project_entity_info = asset_cache.project_entity_info
    if project_entity_info is None or not isinstance(project_entity_info, GeometryEntityInfo):
        return

    entity_info_dict = json_data.get("private_attribute_asset_cache", {}).get(
        "project_entity_info", None
    )
    if not isinstance(entity_info_dict, dict):
        return

    grouped_bodies = entity_info_dict.get("grouped_bodies", None)
    if not isinstance(grouped_bodies, list):
        return

    body_group_tag = entity_info_dict.get("body_group_tag") or project_entity_info.body_group_tag
    body_attribute_names = (
        entity_info_dict.get("body_attribute_names") or project_entity_info.body_attribute_names
    )
    selected_group_index = None
    if (
        isinstance(body_group_tag, str)
        and isinstance(body_attribute_names, list)
        and body_group_tag in body_attribute_names
    ):
        selected_group_index = body_attribute_names.index(body_group_tag)

    cs_mgr = CoordinateSystemManager._from_status(  # pylint: disable=protected-access
        status=asset_cache.coordinate_system_status
    )

    identity_matrix_row_major = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    for i_group, body_groups in enumerate(grouped_bodies):
        if not isinstance(body_groups, list):
            continue
        for body_group in body_groups:
            if not isinstance(body_group, dict):
                continue

            # For the selected grouping, inject only the transformation matrix.
            # The mesher only needs the final 3x4 matrix, not the intermediate parameters.
            if selected_group_index is None or i_group != selected_group_index:
                # For non-selected groupings, remove any existing transformation data
                body_group.pop("transformation", None)
                continue

            entity_type = body_group.get("private_attribute_entity_type_name")
            entity_id = body_group.get("private_attribute_id")

            matrix = None
            if isinstance(entity_type, str) and isinstance(entity_id, str):
                matrix_nd = cs_mgr._get_matrix_for_entity_key(  # pylint: disable=protected-access
                    entity_type=entity_type, entity_id=entity_id
                )
                if matrix_nd is not None:
                    matrix = matrix_nd.flatten().tolist()

            # Only include the matrix - no redundant transformation parameters
            body_group["transformation"] = {
                "private_attribute_matrix": identity_matrix_row_major if matrix is None else matrix
            }


def filter_simulation_json(input_params: SimulationParams, mesh_units):
    """
    Filter the simulation JSON to only include the GAI surface meshing parameters.
    """

    # Get the JSON from the input_params
    json_data = input_params.model_dump(mode="json", exclude_none=True)

    _inject_body_group_transformations_for_mesher(
        json_data=json_data, input_params=input_params, mesh_unit=mesh_units
    )

    # Generate whitelist based on simulation context
    whitelist = _get_gai_setting_whitelist(input_params)

    # Filter the JSON to only include the GAI surface meshing parameters
    filtered_json = _traverse_and_filter(json_data, whitelist)

    return filtered_json


@preprocess_input
# pylint: disable=unused-argument
def get_surface_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.
    """
    ensure_meshing_is_specified(input_params)

    if not input_params.private_attribute_asset_cache.use_geometry_AI:
        if using_snappy(input_params):
            return snappy_mesher_json(input_params)
        if isinstance(input_params.meshing, MeshingParams):
            return legacy_mesher_json(input_params)
        raise Flow360TranslationError(
            f"translation for {type(input_params.meshing)} not implemented.",
            None,
            ["meshing"],
        )

    # === GAI mode ===
    # Just do a filtering of the input_params's JSON
    return filter_simulation_json(input_params, mesh_units)
