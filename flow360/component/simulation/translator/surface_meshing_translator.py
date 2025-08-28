"""Surface meshing parameter translator."""

from copy import deepcopy
from typing import List

import numpy as np

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
    SnappySurfaceMeshingParams,
)
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
    SnappyRegionRefinement,
    SnappySurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box, Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
)
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
        return {"type": "aniso", "method": "height", "value": obj.method.value.value.item()}

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
def SurfaceRefinement_to_faces(obj: SurfaceRefinement, global_max_edge_length):
    """
    Translate SurfaceRefinement to faces.

    """
    return {
        "maxEdgeLength": (
            obj.max_edge_length.value.item()
            if obj.max_edge_length is not None
            else global_max_edge_length.value.item()
        ),
    }


def apply_SnappyBodyRefinement(refinement: SnappyBodyRefinement, translated):
    applicable_bodies = [entity.body_name for entity in refinement.entities]
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_bodies:
            if refinement.gap_resolution is not None:
                body["gap"] = refinement.gap_resolution.value.item()
            if refinement.proximity_spacing is not None:
                body["gapSpacingReduction"] = refinement.proximity_spacing.value.item()
            if refinement.min_spacing is not None:
                body["spacing"]["min"] = refinement.min_spacing.value.item()
            if refinement.max_spacing is not None:
                body["spacing"]["max"] = refinement.max_spacing.value.item()


def get_applicable_regions_dict(refinement_regions):
    applicable_regions = {}
    if refinement_regions:
        for entity in refinement_regions.stored_entities:
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
    refinement: SnappySurfaceEdgeRefinement, translated, defaults
):
    edges = {"includedAngle": refinement.included_angle.to("degree").value.item()}
    if refinement.min_elem is not None:
        edges["minElem"] = refinement.min_elem
    if refinement.min_len is not None:
        edges["minLen"] = refinement.min_len.value.item()
    if refinement.retain_on_smoothing is not None:
        edges["retainOnSmoothing"] = refinement.retain_on_smoothing
    if refinement.spacing is None:
        edges["edgeSpacing"] = defaults.min_spacing.value.item()
    elif isinstance(refinement.spacing, List):
        edges["edgeSpacing"] = [
            [dist.value.item(), spac.value.item()]
            for (dist, spac) in zip(refinement.distances, refinement.spacing)
        ]
    else:
        edges["edgeSpacing"] = refinement.spacing.value.item()
    applicable_bodies = (
        [entity.body_name for entity in refinement.bodies] if refinement.bodies is not None else []
    )
    applicable_regions = get_applicable_regions_dict(refinement_regions=refinement.regions)
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_bodies or (
            body["bodyName"] in applicable_regions and applicable_regions[body["bodyName"]] is None
        ):
            body["edges"] = edges
        if body["bodyName"] in applicable_regions:
            for region in body.get("regions", []):
                if region["patchName"] in applicable_regions[body["bodyName"]]:
                    region["edges"] = edges


def apply_SnappyRegionRefinement(refinement: SnappyRegionRefinement, translated):
    applicable_regions = applicable_regions = get_applicable_regions_dict(
        refinement_regions=refinement.entities
    )
    for body in translated["geometry"]["bodies"]:
        if body["bodyName"] in applicable_regions:
            for region in body.get("regions", []):
                if region["patchName"] in applicable_regions[body["bodyName"]]:
                    if refinement.proximity_spacing is not None:
                        region["gapSpacingReduction"] = refinement.proximity_spacing.value.item()

                    region["spacing"] = {
                        "min": refinement.min_spacing.value.item(),
                        "max": refinement.max_spacing.value.item(),
                    }


def apply_UniformRefinement_w_snappy(refinement: UniformRefinement, translated):
    if "refinementVolumes" not in translated["geometry"]:
        translated["geometry"]["refinementVolumes"] = []

    for volume in refinement.entities.stored_entities:
        volume_body = {"spacing": refinement.spacing.value.item(), "name": volume.name}
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
                f"Volume of type {type(volume)} cannot be used with Snappy."
            )

        translated["geometry"]["refinementVolumes"].append(volume_body)


def legacy_mesher_json(input_params: SimulationParams):
    """
    Get JSON for surface meshing.

    """
    translated = {}
    # pylint: disable=duplicate-code
    if input_params.meshing is None:
        raise Flow360TranslationError(
            "meshing not specified.",
            None,
            ["meshing"],
        )

    if isinstance(input_params.meshing, ModularMeshingWorkflow) and isinstance(
        input_params.meshing.surface_meshing, SnappySurfaceMeshingParams
    ):

        surface_meshing_params = input_params.meshing.surface_meshing
        # extract geometry information in body: {patch0, ...} format
        bodies = {}
        for face_id in input_params.private_attribute_asset_cache.project_entity_info.face_ids:
            solid = face_id.split("::")
            if solid[0] not in bodies:
                bodies[solid[0]] = set()
            if len(solid) == 2:
                bodies[solid[0]].add(solid[1])

        # Fill with defaults
        common_defaults = {
            "gap": surface_meshing_params.defaults.gap_resolution.value.item(),
            "spacing": {
                "min": surface_meshing_params.defaults.min_spacing.value.item(),
                "max": surface_meshing_params.defaults.max_spacing.value.item(),
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
        # apply refinements
        for refinement in surface_meshing_params.refinements:
            if isinstance(refinement, SnappyBodyRefinement):
                apply_SnappyBodyRefinement(refinement, translated)
            elif isinstance(refinement, SnappySurfaceEdgeRefinement):
                apply_SnappySurfaceEdgeRefinement(
                    refinement, translated, surface_meshing_params.defaults
                )
            elif isinstance(refinement, SnappyRegionRefinement):
                apply_SnappyRegionRefinement(refinement, translated)
            elif isinstance(refinement, UniformRefinement):
                apply_UniformRefinement_w_snappy(refinement, translated)
            else:
                raise Flow360TranslationError(
                    f"Refinement of type {type(refinement)} cannot be used with Snappy."
                )

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
                    "nSolveIter": snap_controls.n_solve_iter,
                    "nRelaxIter": snap_controls.n_relax_iter,
                    "nFeatureSnapIter": snap_controls.n_feature_snap_iter,
                    "multiRegionFeatureSnap": snap_controls.multi_region_feature_snap,
                    "strictRegionSnap": snap_controls.strict_region_snap,
                },
            },
            "meshQuality": {
                "maxNonOrtho": (
                    quality_settings.max_non_ortho.to("degree").value.item()
                    if quality_settings.max_non_ortho is not None
                    else 180
                ),
                "maxBoundarySkewness": (
                    quality_settings.max_boundary_skewness.to("degree").value.item()
                    if quality_settings.max_boundary_skewness is not None
                    else -1
                ),
                "maxInternalSkewness": (
                    quality_settings.max_internal_skewness.to("degree").value.item()
                    if quality_settings.max_internal_skewness is not None
                    else -1
                ),
                "maxConcave": (
                    quality_settings.max_concave.to("degree").value.item()
                    if quality_settings.max_concave is not None
                    else 180
                ),
                "minVol": (
                    quality_settings.min_vol if quality_settings.min_vol is not None else -1e30
                ),
                "minTetQuality": (
                    quality_settings.min_tet_quality
                    if quality_settings.min_tet_quality is not None
                    else -1e30
                ),
                "minArea": (
                    quality_settings.min_area.value.item()
                    if quality_settings.min_area is not None
                    else -1
                ),
                "minTwist": (
                    quality_settings.min_twist if quality_settings.min_twist is not None else -2
                ),
                "minDeterminant": (
                    quality_settings.min_determinant
                    if quality_settings.min_determinant is not None
                    else -1e5
                ),
                "minVolRatio": (
                    quality_settings.min_vol_ratio
                    if quality_settings.min_vol_ratio is not None
                    else 0
                ),
                "minFaceWeight": (
                    quality_settings.min_face_weight
                    if quality_settings.min_face_weight is not None
                    else 0
                ),
                "minTriangleTwist": (
                    quality_settings.min_triangle_twist
                    if quality_settings.min_triangle_twist is not None
                    else -1
                ),
                "nSmoothScale": (
                    quality_settings.n_smooth_scale
                    if quality_settings.n_smooth_scale is not None
                    else 0
                ),
                "errorReduction": (
                    quality_settings.error_reduction
                    if quality_settings.error_reduction is not None
                    else 0
                ),
                "minVolCollapseRatio": (
                    quality_settings.min_vol_collapse_ratio
                    if quality_settings.min_vol_collapse_ratio is not None
                    else 0
                ),
            },
        }
        # smoothing settings
        smoothing_settings = surface_meshing_params.smooth_controls

        if smoothing_settings is not None:
            translated["smoothingControls"] = {
                "lambda": (
                    smoothing_settings.lambda_factor
                    if smoothing_settings.lambda_factor is not None
                    else 0
                ),
                "mu": (
                    smoothing_settings.mu_factor if smoothing_settings.mu_factor is not None else 0
                ),
                "iter": (
                    smoothing_settings.iterations
                    if smoothing_settings.iterations is not None
                    else 0
                ),
            }
            if smoothing_settings.included_angle is None or np.isclose(
                smoothing_settings.included_angle.to("degree").value.item(), 0
            ):
                translated["smoothingControls"]["includedAngle"] = None
            else:
                translated["smoothingControls"]["includedAngle"] = (
                    smoothing_settings.included_angle.to("degree").value.item()
                )

            if smoothing_settings.min_elem is not None:
                translated["smoothingControls"]["minElem"] = smoothing_settings.min_elem
            if smoothing_settings.min_len is not None:
                translated["smoothingControls"]["minLen"] = smoothing_settings.min_len.value.item()

        # bounding box
        bounding_box = surface_meshing_params.bounding_box

        if bounding_box is not None:
            translated["boundingBox"] = {
                "min": {
                    "x": bounding_box.center[0].value.item()
                    - (bounding_box.size[0].value.item() / 2),
                    "y": bounding_box.center[1].value.item()
                    - (bounding_box.size[1].value.item() / 2),
                    "z": bounding_box.center[2].value.item()
                    - (bounding_box.size[2].value.item() / 2),
                },
                "max": {
                    "x": bounding_box.center[0].value.item()
                    + (bounding_box.size[0].value.item() / 2),
                    "y": bounding_box.center[1].value.item()
                    + (bounding_box.size[1].value.item() / 2),
                    "z": bounding_box.center[2].value.item()
                    + (bounding_box.size[2].value.item() / 2),
                },
            }
        # points in mesh
        zones = surface_meshing_params.zones
        if zones is not None:
            translated["locationInMesh"] = {
                zone.name: [point.value.item() for point in zone.point_in_mesh] for zone in zones
            }

        # cad is fluid
        if surface_meshing_params.cad_is_fluid:
            translated["cadIsFluid"] = True

    elif isinstance(input_params.meshing, MeshingParams):
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
            use_sub_item_as_key=True,
        )

        ##:: >> Step 5.1: Apply default_max_edge_length to faces that are not explicitly specified
        assert input_params.private_attribute_asset_cache.project_entity_info is not None
        assert isinstance(
            input_params.private_attribute_asset_cache.project_entity_info, GeometryEntityInfo
        )

        for face_id in input_params.private_attribute_asset_cache.project_entity_info.face_ids:
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
    else:
        raise Flow360TranslationError(
            f"translation for {type(input_params.meshing)} not implemented.",
            None,
            ["meshing"],
        )
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


GAI_SETTING_WHITELIST = {
    "meshing": {
        "defaults": {
            "surface_max_edge_length": None,
            "curvature_resolution_angle": None,
            "surface_edge_growth_rate": None,
            "geometry_accuracy": None,
            "surface_max_aspect_ratio": None,
            "surface_max_adaptation_iterations": None,
        },
        "refinements": _get_surface_refinements,
    },
    "private_attribute_asset_cache": {
        "project_entity_info": {
            "face_group_tag": None,
            "face_attribute_names": None,
            "grouped_faces": None,
            "body_group_tag": None,
            "body_attribute_names": None,
            "grouped_bodies": None,
        }
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


def filter_simulation_json(input_params: SimulationParams):
    """
    Filter the simulation JSON to only include the GAI surface meshing parameters.
    """

    # Get the JSON from the input_params
    json_data = input_params.model_dump(mode="json", exclude_none=True)

    # Filter the JSON to only include the GAI surface meshing parameters
    filtered_json = _traverse_and_filter(json_data, GAI_SETTING_WHITELIST)

    return filtered_json


@preprocess_input
# pylint: disable=unused-argument
def get_surface_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.
    """
    if not input_params.private_attribute_asset_cache.use_geometry_AI:
        return legacy_mesher_json(input_params)

    # === GAI mode ===
    input_params.private_attribute_asset_cache.project_entity_info.compute_transformation_matrices()
    # Just do a filtering of the input_params's JSON
    return filter_simulation_json(input_params)
