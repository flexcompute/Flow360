"""Volume meshing parameter translator."""

from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import BoundaryLayer
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_attribute_from_first_instance,
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
)


def unifrom_refinement_translator(obj: SurfaceEdgeRefinement):
    """
    Translate UniformRefinement.

    """
    return {"spacing": obj.spacing.value.item()}


def _entitity_info_seralizer(entity_obj):
    if isinstance(entity_obj, Cylinder):
        return {
            "type": "cylinder",
            "radius": entity_obj.outer_radius.value.item(),
            "length": entity_obj.height.value.item(),
            "axis": list(entity_obj.axis),
            "center": list(entity_obj.center.value),
        }
    return {}


@preprocess_input
# pylint: disable=unused-argument
def get_volume_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.

    """
    translated = {}

    # >>  Step 1:  Get refinementFactor
    translated["refinementFactor"] = input_params.meshing.refinement_factor

    # >> Step 2: Get volume refinements
    translated["refinement"] = translate_setting_and_apply_to_all_entities(
        input_params.meshing.refinements,
        UniformRefinement,
        unifrom_refinement_translator,
        to_list=True,
        entity_injection_func=_entitity_info_seralizer,
    )

    # >> Step 3: Get volumetric global settings
    # firstLayerThickness can be locally overridden
    translated["volume"] = {}
    translated["volume"]["firstLayerThickness"] = get_attribute_from_first_instance(
        input_params.meshing.refinements,
        BoundaryLayer,
        "first_layer_thickness",
        use_empty_entities=True,
    ).value.item()
    translated["volume"]["growthRate"] = get_attribute_from_first_instance(
        input_params.meshing.refinements,
        BoundaryLayer,
        "growth_rate",
        use_empty_entities=True,
    )

    return translated
