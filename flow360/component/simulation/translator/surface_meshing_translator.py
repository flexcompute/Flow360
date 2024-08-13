"""Surface meshing parameter translator."""

from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_global_setting_from_per_item_setting,
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
        return {"type": "aniso", "method": "aspectRatio", "value": obj.method.value.value.item()}

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

    if input_params.meshing.refinements is None:
        log.info("No `refinements` found in the input. Skipping translation.")
        raise Flow360TranslationError(
            "No `refinements` found in the input",
            input_value=None,
            location=["meshing", "refinements"],
        )

    ##:: >>  Step 1:  Get global maxEdgeLength [REQUIRED]
    ##~~ On SimulationParam side
    ##~~ If there is a SurfaceRefinement with empty entities, it is considered as global setting.
    ##~~ If there is no such SurfaceRefinement, then we enforce that all surfaces specified the max_edge_length.
    ##~~
    ##~~ On the translator side
    ##~~ we get the global max_edge_length from the first instance of SurfaceRefinement with empty entities.
    ##~~ If there is no such SurfaceRefinement, we just pick a random one to make surface meshing schema happy.

    # Get from the first instance of SurfaceRefinement with empty entities
    global_max_edge_length = get_global_setting_from_per_item_setting(
        input_params.meshing.refinements,
        SurfaceRefinement,
        "max_edge_length",
        allow_get_from_first_instance_as_fallback=True,
    )
    translated["maxEdgeLength"] = global_max_edge_length.value.item()

    ##:: >> Step 2: Get curvatureResolutionAngle [REQUIRED]
    ##~~ `curvature_resolution_angle` can not be overridden per face.
    ##~~ This can only appear on `SurfaceRefinement` instance without entities.
    global_curvature_resolution_angle = get_global_setting_from_per_item_setting(
        input_params.meshing.refinements,
        SurfaceRefinement,
        "curvature_resolution_angle",
        allow_get_from_first_instance_as_fallback=False,  # No per item override allowed
    )
    translated["curvatureResolutionAngle"] = global_curvature_resolution_angle.to(
        "degree"
    ).value.item()

    ##:: >> Step 3: Get growthRate [REQUIRED]
    ##~~ Same logic as `curvature_resolution_angle`
    surface_layer_growth_rate = get_global_setting_from_per_item_setting(
        input_params.meshing.refinements,
        SurfaceEdgeRefinement,
        "growth_rate",
        allow_get_from_first_instance_as_fallback=False,  # No per item override allowed
    )
    translated["growthRate"] = surface_layer_growth_rate

    ##:: >> Step 4: Get edges [OPTIONAL]
    edge_config = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements,
        SurfaceEdgeRefinement,
        translation_func=SurfaceEdgeRefinement_to_edges,
    )
    if edge_config != {}:
        translated["edges"] = edge_config

    ##:: >> Step 5: Get faces [OPTIONAL]
    face_config = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements,
        SurfaceRefinement,
        translation_func=SurfaceRefinement_to_faces,
        translation_func_global_max_edge_length=global_max_edge_length,
    )
    if face_config != {}:
        translated["faces"] = face_config

    return translated
