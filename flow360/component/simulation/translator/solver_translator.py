"""Flow360 solver setting parameter translator."""

from copy import deepcopy

from flow360.component.simulation.framework.multi_constructor_model_base import (
    _model_attribute_unlock,
)
from typing import Type, Union

from flow360.component.simulation.framework.unique_list import UniqueAliasedStringList
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.models.volume_models import BETDisk, Fluid, Rotation
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Isosurface,
    IsosurfaceOutput,
    Probe,
    ProbeOutput,
    Slice,
    SliceOutput,
    SurfaceIntegralOutput,
    SurfaceList,
    SurfaceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    VolumeOutput,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.translator.utils import (
    convert_tuples_to_lists,
    get_attribute_from_first_instance,
    get_global_setting_from_per_item_setting,
    has_instance_in_list,
    merge_unique_item_lists,
    preprocess_input,
    remove_units_in_dict,
    replace_dict_key,
    translate_setting_and_apply_to_all_entities,
)
from flow360.component.simulation.unit_system import LengthType


def dump_dict(input_params):
    """Dumping param/model to dictionary."""
    return input_params.model_dump(by_alias=True, exclude_none=True)


def init_non_average_output(
    base: dict,
    obj_list,
    class_type: Union[SliceOutput, IsosurfaceOutput, VolumeOutput, SurfaceOutput],
):
    """Initialize the common output attribute for non-average output."""
    has_average_capability = class_type.__name__.endswith(("VolumeOutput", "SurfaceOutput"))
    if has_average_capability:
        base["computeTimeAverages"] = False

    base["animationFrequency"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency", allow_first_instance_as_dummy=True
    )
    base["animationFrequencyOffset"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency_offset", allow_first_instance_as_dummy=True
    )
    return base


def init_average_output(
    base: dict,
    obj_list,
    class_type: Union[TimeAverageVolumeOutput, TimeAverageSurfaceOutput],
):
    """Initialize the common output attribute for average output."""
    base["computeTimeAverages"] = True
    base["animationFrequencyTimeAverage"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency", allow_first_instance_as_dummy=True
    )
    base["animationFrequencyTimeAverageOffset"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency_offset", allow_first_instance_as_dummy=True
    )
    base["startAverageIntegrationStep"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "start_step", allow_first_instance_as_dummy=True
    )
    return base


def init_output_base(obj_list, class_type: Type):
    """Initialize the common output attribute."""

    base = {"outputFields": []}
    output_format = get_global_setting_from_per_item_setting(
        obj_list, class_type, "output_format", allow_first_instance_as_dummy=True
    )
    assert output_format is not None
    if output_format == "both":
        output_format = "paraview,tecplot"
    base["outputFormat"] = output_format

    is_average = class_type.__name__.startswith("TimeAverage")
    if is_average:
        base = init_average_output(base, obj_list, class_type)
    else:
        base = init_non_average_output(base, obj_list, class_type)
    return base


def add_unused_output_settings_for_comparison(output_dict: dict):
    """
    Add unused output settings for easier debugging/comparsions.
    """
    for freq_key in ["animationFrequencyTimeAverage", "animationFrequency"]:
        if freq_key not in output_dict:
            output_dict[freq_key] = -1
    for offset_key in ["animationFrequencyTimeAverageOffset", "animationFrequencyOffset"]:
        if offset_key not in output_dict:
            output_dict[offset_key] = 0
    if "startAverageIntegrationStep" not in output_dict:
        output_dict["startAverageIntegrationStep"] = -1
    return output_dict


def rotation_entity_info_serializer(volume):
    """Rotation entity serializer"""
    return {
        "referenceFrame": {
            "axisOfRotation": list(volume.axis),
            "centerOfRotation": list(volume.center.value),
        },
    }


def rotation_translator(model: Rotation):
    """Rotation translator"""
    volume_zone = {
        "modelType": "FluidDynamics",
        "referenceFrame": {},
    }
    if model.parent_volume:
        volume_zone["referenceFrame"]["parentVolumeName"] = model.parent_volume.name
    spec = dump_dict(model)["spec"]
    if isinstance(spec, str):
        volume_zone["referenceFrame"]["thetaRadians"] = spec
    elif spec.get("units", "") == "flow360_angular_velocity_unit":
        volume_zone["referenceFrame"]["omegaRadians"] = spec["value"]
    return volume_zone


def merge_output_fields(output_model: SurfaceOutput, shared_output_fields: UniqueAliasedStringList):
    """Get merged output fields"""
    if shared_output_fields is None:
        return {"outputFields": output_model.output_fields.items}
    return {
        "outputFields": merge_unique_item_lists(
            output_model.output_fields.items, shared_output_fields.items
        )
    }


def inject_slice_info(entity: Slice):
    """inject entity info"""
    return {
        "sliceOrigin": list(entity.origin.value),
        "sliceNormal": list(entity.normal),
    }


def inject_isosurface_info(entity: Isosurface):
    """inject entity info"""
    return {
        "surfaceField": entity.field,
        "surfaceFieldMagnitude": entity.iso_value,
    }


def inject_probe_info(entity: Probe):
    """inject entity info"""
    return {
        "monitor_locations": [item.value.tolist() for item in entity.locations],
    }


def inject_surface_list_info(entity: SurfaceList):
    """inject entity info"""
    # pylint: disable=fixme
    # TODO: Let's use surface full name"""
    return {
        "surfaces": [surface.name for surface in entity.entities.stored_entities],
    }


def translate_volume_output(
    output_params: list, volume_output_class: Union[VolumeOutput, TimeAverageVolumeOutput]
):
    """Translate volume output settings."""
    volume_output = init_output_base(output_params, volume_output_class)
    # Get outputFields
    volume_output.update(
        {
            "outputFields": get_attribute_from_first_instance(
                output_params, volume_output_class, "output_fields"
            ).model_dump()["items"],
        }
    )
    return volume_output


def translate_surface_output(
    output_params: list, surface_output_class: Union[SurfaceOutput, TimeAverageSurfaceOutput]
):
    """Translate surface output settings."""
    # TODO: Add all boundaries if entities is None after retriving the global settings
    surface_output = init_output_base(output_params, surface_output_class)
    shared_output_fields = get_global_setting_from_per_item_setting(
        output_params,
        surface_output_class,
        "output_fields",
        allow_first_instance_as_dummy=False,
        return_none_when_no_global_found=True,
    )
    surface_output["surfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        surface_output_class,
        translation_func=merge_output_fields,
        to_list=False,
        shared_output_fields=shared_output_fields,
    )
    surface_output["writeSingleFile"] = get_global_setting_from_per_item_setting(
        output_params, surface_output_class, "write_single_file", allow_first_instance_as_dummy=True
    )
    return surface_output


def translate_slice_isosurface_output(
    output_params: list,
    output_class: Union[SliceOutput, IsosurfaceOutput],
    entities_name_key: str,
    injection_function,
):
    """Translate slice or isosurface output settings."""
    translated_output = init_output_base(output_params, output_class)
    shared_output_fields = get_global_setting_from_per_item_setting(
        output_params,
        output_class,
        "output_fields",
        allow_first_instance_as_dummy=False,
        return_none_when_no_global_found=True,
    )
    translated_output[entities_name_key] = translate_setting_and_apply_to_all_entities(
        output_params,
        output_class,
        translation_func=merge_output_fields,
        to_list=False,
        shared_output_fields=shared_output_fields,
        entity_injection_func=injection_function,
    )
    return translated_output


def translate_monitor_output(output_params: list, monitor_type, injection_function):
    """Translate monitor output settings."""
    translated_output = {"outputFields": []}
    shared_output_fields = get_global_setting_from_per_item_setting(
        output_params,
        monitor_type,
        "output_fields",
        allow_first_instance_as_dummy=False,
        return_none_when_no_global_found=True,
    )
    translated_output["monitors"] = translate_setting_and_apply_to_all_entities(
        output_params,
        monitor_type,
        translation_func=merge_output_fields,
        to_list=False,
        shared_output_fields=shared_output_fields,
        entity_injection_func=injection_function,
    )
    return translated_output


def merge_monitor_output(probe_output: dict, integral_output: dict):
    """Merge probe and surface integral output."""
    if probe_output == {}:
        return integral_output
    if integral_output == {}:
        return probe_output

    for integral_output_name, integral_output_value in integral_output["monitors"].items():
        assert integral_output_name not in probe_output["monitors"]
        probe_output["monitors"][integral_output_name] = integral_output_value
    return probe_output


def translate_acoustic_output(output_params: list):
    """Translate acoustic output settings."""
    aeroacoustic_output = {}
    for output in output_params:
        if isinstance(output, AeroAcousticOutput):
            aeroacoustic_output["observers"] = [item.value.tolist() for item in output.observers]
            aeroacoustic_output["writePerSurfaceOutput"] = output.write_per_surface_output
            aeroacoustic_output["patchType"] = output.patch_type
            return aeroacoustic_output
    return None


# pylint: disable=too-many-branches
def translate_output(input_params: SimulationParams, translated: dict):
    """Translate output settings."""
    outputs = input_params.outputs

    if outputs is None:
        return translated
    ##:: Step1: Get translated["volumeOutput"]
    volume_output = {}
    volume_output_average = {}
    if has_instance_in_list(outputs, VolumeOutput):
        volume_output = translate_volume_output(outputs, VolumeOutput)
    if has_instance_in_list(outputs, TimeAverageVolumeOutput):
        volume_output_average = translate_volume_output(outputs, TimeAverageVolumeOutput)
    # Merge
    volume_output.update(**volume_output_average)
    if volume_output:
        translated["volumeOutput"] = add_unused_output_settings_for_comparison(volume_output)

    ##:: Step2: Get translated["surfaceOutput"]
    surface_output = {}
    surface_output_average = {}
    if has_instance_in_list(outputs, SurfaceOutput):
        surface_output = translate_surface_output(outputs, SurfaceOutput)
    if has_instance_in_list(outputs, TimeAverageSurfaceOutput):
        surface_output_average = translate_surface_output(outputs, TimeAverageSurfaceOutput)
    # Merge
    surface_output.update(**surface_output_average)
    if surface_output:
        translated["surfaceOutput"] = add_unused_output_settings_for_comparison(surface_output)

    ##:: Step3: Get translated["sliceOutput"]
    if has_instance_in_list(outputs, SliceOutput):
        translated["sliceOutput"] = translate_slice_isosurface_output(
            outputs, SliceOutput, "slices", inject_slice_info
        )

    ##:: Step4: Get translated["isoSurfaceOutput"]
    if has_instance_in_list(outputs, IsosurfaceOutput):
        translated["isoSurfaceOutput"] = translate_slice_isosurface_output(
            outputs, IsosurfaceOutput, "isoSurfaces", inject_isosurface_info
        )

    ##:: Step5: Get translated["monitorOutput"]
    probe_output = {}
    integral_output = {}
    if has_instance_in_list(outputs, ProbeOutput):
        probe_output = translate_monitor_output(outputs, ProbeOutput, inject_probe_info)
    if has_instance_in_list(outputs, SurfaceIntegralOutput):
        integral_output = translate_monitor_output(
            outputs, SurfaceIntegralOutput, inject_surface_list_info
        )
    # Merge
    if probe_output or integral_output:
        translated["monitorOutput"] = merge_monitor_output(probe_output, integral_output)

    ##:: Step6: Get translated["aeroacousticOutput"]
    if has_instance_in_list(outputs, AeroAcousticOutput):
        translated["aeroacousticOutput"] = translate_acoustic_output(outputs)
    return translated


# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
@preprocess_input
def get_solver_json(
    input_params: SimulationParams,
    # pylint: disable=no-member
    mesh_unit: LengthType.Positive,
):
    """
    Get the solver json from the simulation parameters.
    """

    translated = {}
    ##:: Step 1: Get geometry:
    geometry = remove_units_in_dict(dump_dict(input_params.reference_geometry))
    ml = geometry["momentLength"]
    translated["geometry"] = {
        "momentCenter": list(geometry["momentCenter"]),
        "momentLength": list(ml) if isinstance(ml, tuple) else [ml, ml, ml],
        "refArea": geometry["area"],
    }
    ##:: Step 2: Get freestream
    op = input_params.operating_condition
    translated["freestream"] = {
        "alphaAngle": op.alpha.to("degree").v.item(),
        "betaAngle": op.beta.to("degree").v.item(),
        "Mach": op.velocity_magnitude.v.item(),
        "Temperature": op.thermal_state.temperature.to("K").v.item(),
        "muRef": op.thermal_state.mu_ref(mesh_unit),
    }
    if "reference_velocity_magnitude" in op.model_fields.keys() and op.reference_velocity_magnitude:
        translated["freestream"]["MachRef"] = op.reference_velocity_magnitude.v.item()

    ##:: Step 3: Get boundaries
    translated["boundaries"] = {}
    for model in input_params.models:
        if isinstance(model, (Freestream, SlipWall, SymmetryPlane, Wall)):
            spec = dump_dict(model)
            spec.pop("surfaces")
            spec.pop("name", None)
            if isinstance(model, Wall):
                spec.pop("useWallFunction")
                spec["type"] = "WallFunction" if model.use_wall_function else "NoSlipWall"
                if model.heat_spec:
                    spec.pop("heat_spec")
                    # pylint: disable=fixme
                    # TODO: implement
            for surface in model.entities.stored_entities:
                if surface.private_attribute_full_name is None:
                    with _model_attribute_unlock(surface, "private_attribute_full_name"):
                        surface.private_attribute_full_name = surface.name
                translated["boundaries"][surface.private_attribute_full_name] = spec

    ##:: Step 4: Get outputs
    # TODO:  add a unit test for this
    # # the below is to ensure output fields if no surfaces are defined
    # output_fields = []
    # surface_outputs = [obj for obj in outputs if isinstance(obj, SurfaceOutput)]
    # if len(surface_outputs) == 1:
    #     if surface_outputs[0].entities is None:
    #         output_fields = surface_outputs[0].output_fields.items

    translated = translate_output(input_params, translated)

    ##:: Step 5: Get timeStepping
    ts = input_params.time_stepping
    if isinstance(ts, Unsteady):
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": ts.steps,
            "orderOfAccuracy": ts.order_of_accuracy,
            "maxPseudoSteps": ts.max_pseudo_steps,
            "timeStepSize": ts.step_size.value.item(),
        }
    elif isinstance(ts, Steady):
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": 1,
            "orderOfAccuracy": ts.order_of_accuracy,
            "maxPseudoSteps": ts.max_steps,
            "timeStepSize": "inf",
        }
    dump_dict(input_params.time_stepping)

    ##:: Step 6: Get solver settings
    for model in input_params.models:
        if isinstance(model, Fluid):
            translated["navierStokesSolver"] = dump_dict(model.navier_stokes_solver)
            replace_dict_key(translated["navierStokesSolver"], "typeName", "modelType")
            translated["turbulenceModelSolver"] = dump_dict(model.turbulence_model_solver)
            replace_dict_key(translated["turbulenceModelSolver"], "typeName", "modelType")
            modeling_constants = translated["turbulenceModelSolver"].get("modelingConstants", None)
            if modeling_constants:
                modeling_constants["C_d"] = modeling_constants.pop("CD", None)
                modeling_constants["C_DES"] = modeling_constants.pop("CDES", None)
                modeling_constants.pop("typeName", None)
                translated["turbulenceModelSolver"]["modelConstants"] = translated[
                    "turbulenceModelSolver"
                ].pop("modelingConstants")

    ##:: Step 7: Get BET and AD lists
    for model in input_params.models:
        if isinstance(model, BETDisk):
            disk_param = convert_tuples_to_lists(remove_units_in_dict(dump_dict(model)))
            replace_dict_key(disk_param, "machNumbers", "MachNumbers")
            replace_dict_key(disk_param, "reynoldsNumbers", "ReynoldsNumbers")
            volumes = disk_param.pop("volumes")
            for extra_attr in ["name", "type"]:
                if extra_attr in disk_param:
                    disk_param.pop(extra_attr)
            for v in volumes["storedEntities"]:
                disk_i = deepcopy(disk_param)
                disk_i["axisOfRotation"] = v["axis"]
                disk_i["centerOfRotation"] = v["center"]
                disk_i["radius"] = v["outerRadius"]
                disk_i["thickness"] = v["height"]
                bet_disks = translated.get("BETDisks", [])
                bet_disks.append(disk_i)
                translated["BETDisks"] = bet_disks

    ##:: Step 8: Get rotation
    if has_instance_in_list(input_params.models, Rotation):
        volume_zones = translated.get("volumeZones", {})
        volume_zones.update(
            translate_setting_and_apply_to_all_entities(
                input_params.models,
                Rotation,
                rotation_translator,
                to_list=False,
                entity_injection_func=rotation_entity_info_serializer,
            )
        )
        translated["volumeZones"] = volume_zones

    ##:: Step 9: Get porous media

    ##:: Step 10: Get heat transfer zones

    ##:: Step 11: Get user defined dynamics
    if input_params.user_defined_dynamics is not None:
        translated["userDefinedDynamics"] = []
        for udd in input_params.user_defined_dynamics:
            udd_dict = dump_dict(udd)
            udd_dict["dynamicsName"] = udd_dict.pop("name")
            if udd.input_boundary_patches is not None:
                udd_dict["inputBoundaryPatches"] = []
                for surface in udd.input_boundary_patches.stored_entities:
                    udd_dict["inputBoundaryPatches"].append(surface.name)
            if udd.output_target is not None:
                udd_dict["outputTargetName"] = udd.output_target.name
            translated["userDefinedDynamics"].append(udd_dict)

    return translated
