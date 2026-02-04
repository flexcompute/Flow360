"""Volume meshing parameter translator."""

from typing import Union

from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    PassiveSpacing,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
    VolumeMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CustomZones,
    MeshSliceOutput,
    RotationCylinder,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
    WindTunnelFarfield,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    SeedpointVolume,
    Sphere,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import inject_slice_info
from flow360.component.simulation.translator.utils import (
    ensure_meshing_is_specified,
    get_global_setting_from_first_instance,
    has_instance_in_list,
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


def cylindrical_refinement_translator(obj: Union[AxisymmetricRefinement, RotationVolume]):
    """
    Translate AxisymmetricRefinement or RotationVolume with Cylinder/AxisymmetricBody entities.
    
    Note: This should not be called for RotationVolume with Sphere entities.
    Use spherical_refinement_translator() for those cases.
    """
    if obj.spacing_axial is None or obj.spacing_radial is None or obj.spacing_circumferential is None:
        raise ValueError(
            "cylindrical_refinement_translator requires all spacing fields to be specified. "
            "For Sphere entities in RotationVolume, use spherical_refinement_translator instead."
        )
    
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


def spherical_refinement_translator(obj: RotationVolume):
    """
    Translate RotationVolume with Sphere entity.
    Sphere only uses spacing_circumferential as maxEdgeLength.
    """
    return {
        "maxEdgeLength": obj.spacing_circumferential.value.item(),
    }


def rotation_volume_translator(obj: RotationVolume, rotor_disk_names: list):
    """Setting translation for RotationVolume."""
    # Check if the entity is a Sphere (uses different spacing fields)
    entity = obj.entities.stored_entities[0]  # Only single entity allowed
    if is_exact_instance(entity, Sphere):
        setting = spherical_refinement_translator(obj)
    else:
        setting = cylindrical_refinement_translator(obj)

    setting["enclosedObjects"] = []
    if obj.enclosed_entities is not None:
        for enclosed_entity in obj.enclosed_entities.stored_entities:
            if is_exact_instance(enclosed_entity, Cylinder):
                if enclosed_entity.name in rotor_disk_names:
                    # Current sliding interface encloses a rotor disk
                    # Then we append the interface name which is hardcoded "rotorDisk-<name>""
                    setting["enclosedObjects"].append("rotorDisk-" + enclosed_entity.name)
                else:
                    # Current sliding interface encloses another sliding interface
                    # Then we append the interface name which is hardcoded "slidingInterface-<name>""
                    setting["enclosedObjects"].append("slidingInterface-" + enclosed_entity.name)
            elif is_exact_instance(enclosed_entity, AxisymmetricBody):
                setting["enclosedObjects"].append("slidingInterface-" + enclosed_entity.name)
            elif is_exact_instance(enclosed_entity, Sphere):
                setting["enclosedObjects"].append("slidingInterface-" + enclosed_entity.name)
            elif is_exact_instance(enclosed_entity, Box):
                setting["enclosedObjects"].append("structuredBox-" + enclosed_entity.name)
            elif is_exact_instance(enclosed_entity, Surface):
                setting["enclosedObjects"].append(enclosed_entity.name)
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


def rotation_volume_entity_injector(
    entity: Union[Cylinder, AxisymmetricBody, Sphere], use_inhouse_mesher: bool
):
    """Injector for Cylinder, AxisymmetricBody, or Sphere entity in RotationVolume."""
    if isinstance(entity, Cylinder):
        data = {
            "name": entity.name,
            "innerRadius": 0 if entity.inner_radius is None else entity.inner_radius.value.item(),
            "outerRadius": entity.outer_radius.value.item(),
            "thickness": entity.height.value.item(),
            "axisOfRotation": list(entity.axis),
            "center": list(entity.center.value),
        }
        if use_inhouse_mesher:
            data["type"] = "Cylinder"
        return data

    if isinstance(entity, AxisymmetricBody):
        data = {
            "name": entity.name,
            "profileCurve": [list(profile_point.value) for profile_point in entity.profile_curve],
            "axisOfRotation": list(entity.axis),
            "center": list(entity.center.value),
        }
        if use_inhouse_mesher:
            data["type"] = "Axisymmetric"
        return data

    if isinstance(entity, Sphere):
        data = {
            "name": entity.name,
            "radius": entity.radius.value.item(),
            "axisOfRotation": list(entity.axis),
            "center": list(entity.center.value),
        }
        if use_inhouse_mesher:
            data["type"] = "Sphere"
        return data

    return {}


def _get_custom_volumes(volume_zones: list):
    """Get translated custom volumes from volume zones."""

    custom_volumes = []
    for zone in volume_zones:
        if isinstance(zone, CustomZones):
            # Extract CustomVolume and SeedpointVolume from CustomZones
            enforce_tetrahedral = getattr(zone, "element_type") == "tetrahedra"
            for custom_volume in zone.entities.stored_entities:
                if isinstance(custom_volume, CustomVolume):
                    volume_dict = {
                        "name": custom_volume.name,
                        "patches": sorted(
                            [surface.name for surface in custom_volume.boundaries.stored_entities]
                        ),
                    }
                    if enforce_tetrahedral:
                        volume_dict["enforceTetrahedralElements"] = True
                    custom_volumes.append(volume_dict)

                if isinstance(custom_volume, SeedpointVolume):
                    custom_volumes.append(
                        {
                            "name": custom_volume.name,
                            "pointInMesh": [
                                coord.value.item() for coord in custom_volume.point_in_mesh
                            ],
                        }
                    )

    if custom_volumes:
        # Sort custom volumes by name
        custom_volumes.sort(key=lambda x: x["name"])
    return custom_volumes


def translate_mesh_slice_fields(
    model: MeshSliceOutput,
):
    """Translate mesh slice type and radius fields."""
    mesh_slice_fields = {}
    mesh_slice_fields["crinkled"] = model.include_crinkled_slices
    if model.cutoff_radius is not None:
        mesh_slice_fields["cutoffRadius"] = model.cutoff_radius.value.item()
    return mesh_slice_fields


# def _get_seedpoint_zones(volume_zones: list):
#     """
#     Get translated seedpoint volumes from volume zones.
#     To be later filled with data from snappyHexMesh.
#     """
#     seedpoint_zones = []
#     for zone in volume_zones:
#         if isinstance(zone, SeedpointVolume):
#             seedpoint_zones.append(
#                 {
#                     "name": zone.name,
#                     "pointInMesh": [coord.value.item() for coord in zone.point_in_mesh],
#                 }
#             )
#     if seedpoint_zones:
#         # Sort custom volumes by name
#         seedpoint_zones.sort(key=lambda x: x["name"])
#     return seedpoint_zones


def translate_mesh_slice_output(
    output_params: list,
    output_class: MeshSliceOutput,
    injection_function,
):
    """Translate mesh slice output settings."""
    translated_output = {}
    translated_output["slices"] = translate_setting_and_apply_to_all_entities(
        output_params,
        output_class,
        translation_func=translate_mesh_slice_fields,
        to_list=False,
        entity_injection_func=injection_function,
    )
    return translated_output


@preprocess_input
# pylint: disable=unused-argument,too-many-branches,too-many-statements,too-many-locals
def get_volume_meshing_json(input_params: SimulationParams, mesh_units):
    """
    Get JSON for surface meshing.

    """
    volume_zones = None
    refinements = None
    refinement_factor = None
    defaults = None
    gap_treatment_strength = None
    sliding_interface_tolerance = None
    planar_face_tolerance = None

    translated = {}

    ensure_meshing_is_specified(input_params)

    if isinstance(input_params.meshing, ModularMeshingWorkflow) and isinstance(
        input_params.meshing.volume_meshing, VolumeMeshingParams
    ):
        volume_zones = input_params.meshing.zones
        refinements = input_params.meshing.volume_meshing.refinements
        refinement_factor = input_params.meshing.volume_meshing.refinement_factor
        defaults = input_params.meshing.volume_meshing.defaults
        gap_treatment_strength = input_params.meshing.volume_meshing.gap_treatment_strength
        planar_face_tolerance = input_params.meshing.volume_meshing.planar_face_tolerance
        sliding_interface_tolerance = (
            input_params.meshing.volume_meshing.sliding_interface_tolerance
        )

    if isinstance(input_params.meshing, MeshingParams):
        volume_zones = input_params.meshing.volume_zones
        refinements = input_params.meshing.refinements
        refinement_factor = input_params.meshing.refinement_factor
        defaults = input_params.meshing.defaults
        gap_treatment_strength = input_params.meshing.gap_treatment_strength
        planar_face_tolerance = input_params.meshing.defaults.planar_face_tolerance
        sliding_interface_tolerance = input_params.meshing.defaults.sliding_interface_tolerance

    outputs = input_params.meshing.outputs

    if volume_zones is None:
        raise Flow360TranslationError(
            "volume_zones cannot be None for volume meshing",
            volume_zones,
            ["meshing", "volume_zones"],
        )

    if refinements is None:
        raise Flow360TranslationError(
            "No `refinements` found in the input",
            refinements,
            ["meshing", "refinements"],
        )

    ##::  Step 1:  Get refinementFactor
    if refinement_factor is None:
        raise Flow360TranslationError(
            "No `refinement_factor` found for volume meshing.",
            None,
            ["meshing", "refinement_factor"],
        )
    translated["refinementFactor"] = refinement_factor

    ##::  Step 2:  Get farfield
    for zone in volume_zones:
        if isinstance(zone, (UserDefinedFarfield, CustomZones)):
            translated["farfield"] = {"type": "user-defined"}
            if hasattr(zone, "domain_type") and zone.domain_type is not None:
                translated["farfield"]["domainType"] = zone.domain_type

        if isinstance(zone, WindTunnelFarfield):
            translated["farfield"] = {"type": "wind-tunnel"}
            if zone.domain_type is not None:
                translated["farfield"]["domainType"] = zone.domain_type
            break

        if isinstance(zone, AutomatedFarfield):
            translated["farfield"] = {
                "planarFaceTolerance": planar_face_tolerance,
                "relativeSize": float(zone.relative_size),
            }
            if zone.method == "quasi-3d-periodic":
                translated["farfield"]["type"] = "quasi-3d"
                translated["farfield"]["periodic"] = {"type": "translational"}
            else:
                translated["farfield"]["type"] = zone.method

            if zone.domain_type is not None:
                translated["farfield"]["domainType"] = zone.domain_type
            break

    if "farfield" not in translated:
        raise Flow360TranslationError(
            "One `AutomatedFarfield` instance should be specified.",
            None,
            ["meshing", "volume_zones"],
        )

    ##:: Step 3: Get volumetric global settings
    translated["volume"] = {}

    if defaults.boundary_layer_first_layer_thickness is None:
        # `first_layer_thickness` can be locally overridden so after completeness check, we can
        # get away with the first instance's value if global one does not exist.
        default_first_layer_thickness = get_global_setting_from_first_instance(
            refinements,
            BoundaryLayer,
            "first_layer_thickness",
        )
    else:
        default_first_layer_thickness = defaults.boundary_layer_first_layer_thickness

    translated["volume"]["firstLayerThickness"] = default_first_layer_thickness.value.item()

    # growthRate can only be global
    translated["volume"]["growthRate"] = defaults.boundary_layer_growth_rate

    if gap_treatment_strength is None:
        if input_params.private_attribute_asset_cache.use_inhouse_mesher:
            gap_treatment_strength = 1.0  # Conservative default for inhouse mesher
        else:
            gap_treatment_strength = 0.0
    translated["volume"]["gapTreatmentStrength"] = gap_treatment_strength

    if input_params.private_attribute_asset_cache.use_inhouse_mesher:
        number_of_boundary_layers = defaults.number_of_boundary_layers
        translated["volume"]["numBoundaryLayers"] = (
            number_of_boundary_layers if number_of_boundary_layers is not None else -1
        )

        if planar_face_tolerance is not None:
            translated["volume"]["planarFaceTolerance"] = planar_face_tolerance

        if sliding_interface_tolerance is not None:
            translated["volume"]["slidingInterfaceTolerance"] = sliding_interface_tolerance

    ##::  Step 4: Get volume refinements (uniform + rotorDisks)
    uniform_refinement_list = translate_setting_and_apply_to_all_entities(
        refinements,
        UniformRefinement,
        uniform_refinement_translator,
        to_list=True,
        entity_injection_func=refinement_entity_injector,
    )
    rotor_disk_refinement = translate_setting_and_apply_to_all_entities(
        refinements,
        AxisymmetricRefinement,
        cylindrical_refinement_translator,
        to_list=True,
        entity_injection_func=rotor_disks_entity_injector,
    )
    structured_box_refinement = translate_setting_and_apply_to_all_entities(
        refinements,
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
        refinements,
        BoundaryLayer,
        boundary_layer_translator,
        to_list=False,
    )

    faces_passive_setting = translate_setting_and_apply_to_all_entities(
        refinements,
        PassiveSpacing,
        passive_spacing_translator,
        to_list=False,
    )

    translated["faces"] = {}
    translated["faces"].update(faces_aniso_setting)
    translated["faces"].update(faces_passive_setting)

    ##::  Step 5: Get sliding interfaces ()
    sliding_interfaces = translate_setting_and_apply_to_all_entities(
        volume_zones,
        RotationVolume,
        rotation_volume_translator,
        to_list=True,
        entity_injection_func=rotation_volume_entity_injector,
        translation_func_rotor_disk_names=rotor_disk_names,
        entity_injection_use_inhouse_mesher=input_params.private_attribute_asset_cache.use_inhouse_mesher,
    )
    sliding_interfaces_cylinders = translate_setting_and_apply_to_all_entities(
        volume_zones,
        RotationCylinder,
        rotation_volume_translator,
        to_list=True,
        entity_injection_func=rotation_volume_entity_injector,
        translation_func_rotor_disk_names=rotor_disk_names,
        entity_injection_use_inhouse_mesher=input_params.private_attribute_asset_cache.use_inhouse_mesher,
    )

    if sliding_interfaces or sliding_interfaces_cylinders:
        translated["slidingInterfaces"] = sliding_interfaces + sliding_interfaces_cylinders

    ##::  Step 6: Get custom volumes
    custom_volumes = _get_custom_volumes(volume_zones)
    if custom_volumes:
        translated["zones"] = custom_volumes

    ##::  Step 7: Get meshing output fields

    mesh_slice_output_configs = [
        (MeshSliceOutput, "meshSliceOutput"),
    ]
    for output_class, output_key in mesh_slice_output_configs:
        if has_instance_in_list(outputs, output_class):
            slice_output = translate_mesh_slice_output(outputs, output_class, inject_slice_info)
            if slice_output:
                translated[output_key] = slice_output

    return translated
