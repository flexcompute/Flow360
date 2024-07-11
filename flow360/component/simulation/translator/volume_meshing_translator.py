"""Volume meshing parameter translator."""

from flow360.component.simulation.meshing_param.face_params import BoundaryLayer
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CylindricalRefinementBase,
    RotationCylinder,
    UniformRefinement,
)
from flow360.component.simulation.primitives import Box, Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_global_setting_from_per_item_setting,
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
)
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360TranslationError


def unifrom_refinement_translator(obj: UniformRefinement):
    """
    Translate UniformRefinement.

    """
    return {"spacing": obj.spacing.value.item()}


def cylindrical_refinement_translator(obj: CylindricalRefinementBase):
    """
    Translate CylindricalRefinementBase. [SlidingInterface + RotorDisks]
    """
    return {
        "spacingAxial": obj.spacing_axial.value.item(),
        "spacingRadial": obj.spacing_radial.value.item(),
        "spacingCircumferential": obj.spacing_circumferential.value.item(),
    }


def rotation_cylinder_translator(obj: RotationCylinder, rotor_disk_names: list):
    """Setting translation for RotationCylinder."""
    setting = cylindrical_refinement_translator(obj)
    setting["enclosedObjects"] = []
    if obj.enclosed_entities is not None:
        for entity in obj.enclosed_entities.stored_entities:
            if is_exact_instance(entity, Cylinder):
                if entity.name in rotor_disk_names:
                    # Current sliding interface encloses a rotor disk
                    # Then we append the interace name which is hardcoded "rotorDisk-<name>""
                    setting["enclosedObjects"].append("rotorDisk-" + entity.name)
                else:
                    # Current sliding interface encloses another sliding interface
                    # Then we append the interace name which is hardcoded "slidingInterface-<name>""
                    setting["enclosedObjects"].append("slidingInterface-" + entity.name)
            elif is_exact_instance(entity, Surface):
                setting["enclosedObjects"].append(entity.name)
    return setting


def refinement_entitity_injector(entity_obj):
    """Injector for UniformRefinement entity [box & cylinder]."""
    if isinstance(entity_obj, Cylinder):
        return {
            "type": "cylinder",
            "radius": entity_obj.outer_radius.value.item(),
            "length": entity_obj.height.value.item(),
            "axis": list(entity_obj.axis),
            "center": list(entity_obj.center.value),
        }
    if isinstance(entity_obj, Box):
        return {
            "type": "box",
            "size": list(entity_obj.size.value),
            "center": list(entity_obj.center.value),
            "axisOfRotation": list(entity_obj.axis_of_rotation),
            "angleOfRotation": entity_obj.angle_of_rotation.to("degree").value.item(),
        }
    return {}


def rotor_disks_entity_injector(entity: Cylinder):
    """Injector for Cylinder entity in AxisymmetricRefinement."""

    return {
        "name": entity.name,
        "innerRadius": 0 if entity.inner_radius is None else entity.inner_radius.value.item(),
        "outerRadius": entity.outer_radius.value.item(),
        "thickness": entity.height.value.item(),
        "axisThrust": list(entity.axis),
        "center": list(entity.center.value),
    }


def rotation_cylinder_entity_injector(entity: Cylinder):
    """Injector for Cylinder entity in RotationCylinder."""
    return {
        "name": entity.name,
        "innerRadius": 0 if entity.inner_radius is None else entity.inner_radius.value.item(),
        "outerRadius": entity.outer_radius.value.item(),
        "thickness": entity.height.value.item(),
        "axisOfRotation": list(entity.axis),
        "center": list(entity.center.value),
    }


@preprocess_input
# pylint: disable=unused-argument
def get_volume_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.

    """
    translated = {}

    if input_params.meshing is None:
        raise Flow360TranslationError(
            "meshing not specified.",
            None,
            ["meshing"],
        )

    if input_params.meshing.volume_zones is None:
        raise Flow360TranslationError(
            "volume_zones cannot be None for volume meshing",
            input_params.meshing.volume_zones,
            ["meshing", "volume_zones"],
        )

    if input_params.meshing.refinements is None:
        raise Flow360TranslationError(
            "No `refinements` found in the input",
            input_params.meshing.refinements,
            ["meshing", "refinements"],
        )

    meshing_params = input_params.meshing

    ##::  Step 1:  Get refinementFactor
    if meshing_params.refinement_factor is None:
        raise Flow360TranslationError(
            "No `refinement_factor` found for volume meshing.",
            None,
            ["meshing", "refinement_factor"],
        )
    translated["refinementFactor"] = meshing_params.refinement_factor

    ##::  Step 2:  Get farfield
    for zone in input_params.meshing.volume_zones:
        if isinstance(zone, AutomatedFarfield):
            translated["farfield"] = {"type": zone.method}
            break

    if "farfield" not in translated:
        raise Flow360TranslationError(
            "One `AutomatedFarfield` instance should be specified.",
            None,
            ["meshing", "volume_zones"],
        )

    ##:: Step 3: Get volumetric global settings
    translated["volume"] = {}
    # firstLayerThickness can be locally overridden so after completeness check, we can
    # get away with the first instance's value.
    translated["volume"]["firstLayerThickness"] = get_global_setting_from_per_item_setting(
        meshing_params.refinements,
        BoundaryLayer,
        "first_layer_thickness",
        allow_get_from_first_instance_as_fallback=True,
    ).value.item()
    # growthRate can only be global so after completeness check, and use same logic as curvature_resolution_angle.
    translated["volume"]["growthRate"] = get_global_setting_from_per_item_setting(
        meshing_params.refinements,
        BoundaryLayer,
        "growth_rate",
        allow_get_from_first_instance_as_fallback=True,
        # `allow_get_from_first_instance_as_fallback` can be true because completeness check is done in Params
    )
    translated["volume"]["gapTreatmentStrength"] = meshing_params.gap_treatment_strength

    ##::  Step 4: Get volume refinements (uniform + rotorDisks)
    uniform_refinement_list = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        UniformRefinement,
        unifrom_refinement_translator,
        to_list=True,
        entity_injection_func=refinement_entitity_injector,
    )
    rotor_disk_refinement = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        AxisymmetricRefinement,
        cylindrical_refinement_translator,
        to_list=True,
        entity_injection_func=rotor_disks_entity_injector,
    )
    if uniform_refinement_list:
        translated["refinement"] = []
        translated["refinement"].extend(uniform_refinement_list)

    rotor_disk_names = []
    if rotor_disk_refinement:
        translated["rotorDisks"] = []
        translated["rotorDisks"].extend(rotor_disk_refinement)
        rotor_disk_names = [item["name"] for item in rotor_disk_refinement]

    ##::  Step 5: Get sliding interfaces ()
    sliding_interfaces = translate_setting_and_apply_to_all_entities(
        meshing_params.volume_zones,
        RotationCylinder,
        rotation_cylinder_translator,
        to_list=True,
        entity_injection_func=rotation_cylinder_entity_injector,
        rotor_disk_names=rotor_disk_names,
    )
    if sliding_interfaces:
        translated["slidingInterfaces"] = sliding_interfaces

    return translated
