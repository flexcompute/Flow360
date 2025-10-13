"""Volume meshing parameter translator."""

from typing import Union

from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    PassiveSpacing,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    AxisymmetricRefinementBase,
    RotationCylinder,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_global_setting_from_first_instance,
    preprocess_input,
    translate_setting_and_apply_to_all_entities,
)
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360TranslationError

# pylint: disable=too-many-locals


def uniform_refinement_translator(obj: UniformRefinement):
    """
    Translate UniformRefinement.

    """
    return {"spacing": obj.spacing.value.item()}


def cylindrical_refinement_translator(obj: AxisymmetricRefinementBase):
    """
    Translate CylindricalRefinementBase. [SlidingInterface + RotorDisks]
    """
    return {
        "spacingAxial": obj.spacing_axial.value.item(),
        "spacingRadial": obj.spacing_radial.value.item(),
        "spacingCircumferential": obj.spacing_circumferential.value.item(),
    }


def box_refinement_translator(obj: StructuredBoxRefinement):
    """
    Translate StructuredBoxRefinement spacings
    """
    return {
        "spacingAxis1": obj.spacing_axis1.value.item(),
        "spacingAxis2": obj.spacing_axis2.value.item(),
        "spacingNormal": obj.spacing_normal.value.item(),
    }


def boundary_layer_translator(obj: BoundaryLayer):
    """
    Translate BoundaryLayer.

    """
    face = {"type": "aniso"}
    if obj.first_layer_thickness is not None:
        face["firstLayerThickness"] = obj.first_layer_thickness.value.item()
    if obj.growth_rate is not None:
        face["growthRate"] = obj.growth_rate
    return face


def passive_spacing_translator(obj: PassiveSpacing):
    """
    Translate BoundaryLayer.

    """
    spacing_type = None

    if obj.type == "projected":
        spacing_type = "projectAnisoSpacing"
    elif obj.type == "unchanged":
        spacing_type = "none"
    else:
        raise ValueError(f"Unknown type: {obj.type} for PassiveSpacing.")

    return {
        "type": spacing_type,
    }


def rotation_volume_translator(obj: RotationVolume, rotor_disk_names: list):
    """Setting translation for RotationVolume."""
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
            elif is_exact_instance(entity, AxisymmetricBody):
                setting["enclosedObjects"].append("slidingInterface-" + entity.name)
            elif is_exact_instance(entity, Box):
                setting["enclosedObjects"].append("structuredBox-" + entity.name)
            elif is_exact_instance(entity, Surface):
                setting["enclosedObjects"].append(entity.name)
    return setting


def refinement_entity_injector(entity_obj):
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


def refinement_entity_box_with_axes_injector(entity_obj: Box):
    """Injector for Box entity in StructuredBoxRefinement."""
    lengths = list(entity_obj.size.value)

    axis1 = entity_obj.axes[0]
    axis2 = entity_obj.axes[1]

    return {
        "name": entity_obj.name,
        "type": "box",
        "lengthAxis1": lengths[0],
        "lengthAxis2": lengths[1],
        "lengthNormal": lengths[2],
        "axis1": list(axis1),
        "axis2": list(axis2),
        "center": list(entity_obj.center.value),
    }


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


def rotation_volume_entity_injector(entity: Union[Cylinder, AxisymmetricBody]):
    """Injector for Cylinder entity in RotationCylinder."""
    if isinstance(entity, Cylinder):
        return {
            "name": entity.name,
            "type": "Cylinder",
            "innerRadius": 0 if entity.inner_radius is None else entity.inner_radius.value.item(),
            "outerRadius": entity.outer_radius.value.item(),
            "thickness": entity.height.value.item(),
            "axisOfRotation": list(entity.axis),
            "center": list(entity.center.value),
        }
    if isinstance(entity, AxisymmetricBody):
        return {
            "name": entity.name,
            "type": "Axisymmetric",
            "profileCurve": [list(profile_point.value) for profile_point in entity.profile_curve],
            "axisOfRotation": list(entity.axis),
            "center": list(entity.center.value),
        }
    return {}


def _get_custom_volumes(volume_zones: list):
    """Get translated custom volumes from volume zones."""
    custom_volumes = []
    for zone in volume_zones:
        if isinstance(zone, CustomVolume):
            custom_volumes.append(
                {
                    "name": zone.name,
                    "patches": sorted(
                        [surface.name for surface in zone.boundaries.stored_entities]
                    ),
                }
            )
    if custom_volumes:
        # Sort custom volumes by name
        custom_volumes.sort(key=lambda x: x["name"])
    return custom_volumes


@preprocess_input
# pylint: disable=unused-argument,too-many-branches,too-many-statements
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
        if isinstance(zone, UserDefinedFarfield):
            translated["farfield"] = {"type": "user-defined"}
            break

        if isinstance(zone, AutomatedFarfield):
            translated["farfield"] = {
                "planarFaceTolerance": meshing_params.defaults.planar_face_tolerance
            }
            if zone.method == "quasi-3d-periodic":
                translated["farfield"]["type"] = "quasi-3d"
                translated["farfield"]["periodic"] = {"type": "translational"}
            else:
                translated["farfield"]["type"] = zone.method
            break

    if "farfield" not in translated:
        raise Flow360TranslationError(
            "One `AutomatedFarfield` instance should be specified.",
            None,
            ["meshing", "volume_zones"],
        )

    ##:: Step 3: Get volumetric global settings
    translated["volume"] = {}

    if meshing_params.defaults.boundary_layer_first_layer_thickness is None:
        # `first_layer_thickness` can be locally overridden so after completeness check, we can
        # get away with the first instance's value if global one does not exist.
        default_first_layer_thickness = get_global_setting_from_first_instance(
            meshing_params.refinements,
            BoundaryLayer,
            "first_layer_thickness",
        )
    else:
        default_first_layer_thickness = meshing_params.defaults.boundary_layer_first_layer_thickness

    translated["volume"]["firstLayerThickness"] = default_first_layer_thickness.value.item()

    # growthRate can only be global
    translated["volume"]["growthRate"] = meshing_params.defaults.boundary_layer_growth_rate

    translated["volume"]["gapTreatmentStrength"] = meshing_params.gap_treatment_strength

    if input_params.private_attribute_asset_cache.use_inhouse_mesher:
        number_of_boundary_layers = meshing_params.defaults.number_of_boundary_layers
        translated["volume"]["numBoundaryLayers"] = (
            number_of_boundary_layers if number_of_boundary_layers is not None else -1
        )

        translated["volume"]["planarFaceTolerance"] = meshing_params.defaults.planar_face_tolerance

    ##::  Step 4: Get volume refinements (uniform + rotorDisks)
    uniform_refinement_list = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        UniformRefinement,
        uniform_refinement_translator,
        to_list=True,
        entity_injection_func=refinement_entity_injector,
    )
    rotor_disk_refinement = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        AxisymmetricRefinement,
        cylindrical_refinement_translator,
        to_list=True,
        entity_injection_func=rotor_disks_entity_injector,
    )
    structured_box_refinement = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        StructuredBoxRefinement,
        box_refinement_translator,
        to_list=True,
        entity_injection_func=refinement_entity_box_with_axes_injector,
    )

    if uniform_refinement_list:
        translated["refinement"] = []
        translated["refinement"].extend(uniform_refinement_list)

    rotor_disk_names = []
    if rotor_disk_refinement:
        translated["rotorDisks"] = []
        translated["rotorDisks"].extend(rotor_disk_refinement)
        rotor_disk_names = [item["name"] for item in rotor_disk_refinement]

    if structured_box_refinement:
        translated["structuredRegions"] = []
        translated["structuredRegions"].extend(structured_box_refinement)

    faces_aniso_setting = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        BoundaryLayer,
        boundary_layer_translator,
        to_list=False,
    )

    faces_passive_setting = translate_setting_and_apply_to_all_entities(
        meshing_params.refinements,
        PassiveSpacing,
        passive_spacing_translator,
        to_list=False,
    )

    translated["faces"] = {}
    translated["faces"].update(faces_aniso_setting)
    translated["faces"].update(faces_passive_setting)

    ##::  Step 5: Get sliding interfaces ()
    sliding_interfaces = translate_setting_and_apply_to_all_entities(
        meshing_params.volume_zones,
        RotationVolume,
        rotation_volume_translator,
        to_list=True,
        entity_injection_func=rotation_volume_entity_injector,
        translation_func_rotor_disk_names=rotor_disk_names,
    )
    sliding_interfaces_cylinders = translate_setting_and_apply_to_all_entities(
        meshing_params.volume_zones,
        RotationCylinder,
        rotation_volume_translator,
        to_list=True,
        entity_injection_func=rotation_volume_entity_injector,
        translation_func_rotor_disk_names=rotor_disk_names,
    )

    if sliding_interfaces or sliding_interfaces_cylinders:
        translated["slidingInterfaces"] = sliding_interfaces + sliding_interfaces_cylinders

    ##::  Step 6: Get custom volumes
    custom_volumes = _get_custom_volumes(meshing_params.volume_zones)
    if custom_volumes:
        translated["zones"] = custom_volumes

    return translated
