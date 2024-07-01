"""Surface meshing parameter translator."""

from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_attribute_from_instance_list,
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
)


# pylint: disable=invalid-name
def SurfaceEdgeRefinement_to_edges(obj: SurfaceEdgeRefinement):
    """
    Translate SurfaceEdgeRefinement to edges.

    """
    if obj.method.type == "height":
        return {"type": "aniso", "method": "height", "value": obj.method.value.value.item()}
    if obj.method.type == "projectAnisoSpacing":
        return {"type": "projectAnisoSpacing"}
    return None


# pylint: disable=invalid-name
def SurfaceRefinement_to_faces(obj: SurfaceRefinement):
    """
    Translate SurfaceRefinement to faces.

    """
    return {
        "maxEdgeLength": obj.max_edge_length.value.item(),
    }


@preprocess_input
# pylint: disable=unused-argument
def get_surface_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.

    """
    translated = {}
    # pylint: disable=fixme
    # TODO: Validations to be implemented:
    # TODO: Backout what is required from the surface meshing JSON definition
    # >> Check Presence:
    # 1. refinements
    # >> Check confliciting multi instances
    # 1. SurfaceRefinement

    # >>  Step 1:  Get maxEdgeLength
    max_edge_length = get_attribute_from_instance_list(
        input_params.meshing.refinements, SurfaceRefinement, "max_edge_length"
    ).value.item()
    translated["maxEdgeLength"] = max_edge_length

    # >> Step 2: Get curvatureResolutionAngle
    curvature_resolution_angle = (
        get_attribute_from_instance_list(
            input_params.meshing.refinements, SurfaceRefinement, "curvature_resolution_angle"
        )
        .to("degree")
        .value.item()
    )
    translated["curvatureResolutionAngle"] = curvature_resolution_angle

    # >> Step 3: Get growthRate
    translated["growthRate"] = input_params.meshing.surface_layer_growth_rate

    # >> Step 4: Get edges
    translated["edges"] = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements, SurfaceEdgeRefinement, SurfaceEdgeRefinement_to_edges
    )

    # >> Step 5: Get faces
    translated["faces"] = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements, SurfaceRefinement, SurfaceRefinement_to_faces
    )

    return translated
