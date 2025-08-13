"""Surface meshing parameter translator."""

from typing import List

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.primitives import Surface
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


@preprocess_input
# pylint: disable=unused-argument
def get_surface_meshing_json(input_params: SimulationParams, mesh_units):
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

    return translated
