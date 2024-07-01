"""Flow360 solver setting parameter translator."""

from copy import deepcopy
from typing import Type, Union

from flow360.component.simulation.framework.unique_list import UniqueAliasedStringList
from flow360.component.simulation.models.material import Sutherland
from flow360.component.simulation.models.surface_models import (
    Freestream,
    HeatFlux,
    Inflow,
    Mach,
    MassFlowRate,
    Outflow,
    Periodic,
    Pressure,
    SlipWall,
    SurfaceModelTypes,
    SymmetryPlane,
    Temperature,
    TotalPressure,
    Wall,
)
from flow360.component.simulation.models.volume_models import (
    BETDisk,
    Fluid,
    PorousMedium,
    Rotation,
)
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
from flow360.component.simulation.primitives import Box
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.translator.utils import (
    _get_key_name,
    convert_tuples_to_lists,
    get_attribute_from_instance_list,
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
    has_average_capability: bool,
):
    """Initialize the common output attribute for non-average output."""
    has_average_capability = class_type.__name__.endswith(("VolumeOutput", "SurfaceOutput"))
    if has_average_capability:
        base["computeTimeAverages"] = False

    base["animationFrequency"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency", allow_get_from_first_instance_as_fallback=True
    )
    base["animationFrequencyOffset"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency_offset", allow_get_from_first_instance_as_fallback=True
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
        obj_list, class_type, "frequency", allow_get_from_first_instance_as_fallback=True
    )
    base["animationFrequencyTimeAverageOffset"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "frequency_offset", allow_get_from_first_instance_as_fallback=True
    )
    base["startAverageIntegrationStep"] = get_global_setting_from_per_item_setting(
        obj_list, class_type, "start_step", allow_get_from_first_instance_as_fallback=True
    )
    return base


def init_output_base(obj_list, class_type: Type, has_average_capability: bool, is_average: bool):
    """Initialize the common output attribute."""

    base = {"outputFields": []}
    output_format = get_global_setting_from_per_item_setting(
        obj_list, class_type, "output_format", allow_get_from_first_instance_as_fallback=True
    )
    assert output_format is not None
    if output_format == "both":
        output_format = "paraview,tecplot"
    base["outputFormat"] = output_format

    if is_average:
        base = init_average_output(base, obj_list, class_type)
    else:
        base = init_non_average_output(
            base,
            obj_list,
            class_type,
            has_average_capability,
        )
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
    return {
        "surfaces": [surface.full_name for surface in entity.entities.stored_entities],
    }


def translate_volume_output(
    output_params: list, volume_output_class: Union[VolumeOutput, TimeAverageVolumeOutput]
):
    """Translate volume output settings."""
    volume_output = init_output_base(
        output_params,
        volume_output_class,
        has_average_capability=True,
        is_average=volume_output_class is TimeAverageVolumeOutput,
    )
    # Get outputFields
    volume_output.update(
        {
            "outputFields": get_attribute_from_instance_list(
                output_params, volume_output_class, "output_fields"
            ).model_dump()["items"],
        }
    )
    return volume_output


def translate_surface_output(
    output_params: list,
    surface_output_class: Union[SurfaceOutput, TimeAverageSurfaceOutput],
    translated: dict,
):
    """Translate surface output settings."""

    assert "boundaries" in translated  # , "Boundaries must be translated before surface output"

    surface_output = init_output_base(
        output_params,
        surface_output_class,
        has_average_capability=True,
        is_average=surface_output_class is TimeAverageSurfaceOutput,
    )
    shared_output_fields = get_global_setting_from_per_item_setting(
        output_params,
        surface_output_class,
        "output_fields",
        allow_get_from_first_instance_as_fallback=False,
        return_none_when_no_global_found=True,
    )
    surface_output["surfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        surface_output_class,
        translation_func=merge_output_fields,
        to_list=False,
        shared_output_fields=shared_output_fields,
    )
    if shared_output_fields is not None:
        # Note: User specified shared output fields for all surfaces. We need to manually add these for surfaces
        # Note: that did not appear in the SurfaceOutput insntances.
        for boundary_name in translated["boundaries"].keys():
            if boundary_name not in surface_output["surfaces"]:
                surface_output["surfaces"][boundary_name] = {
                    "outputFields": shared_output_fields.items
                }
    surface_output["writeSingleFile"] = get_global_setting_from_per_item_setting(
        output_params,
        surface_output_class,
        "write_single_file",
        allow_get_from_first_instance_as_fallback=True,
    )
    return surface_output


def translate_slice_isosurface_output(
    output_params: list,
    output_class: Union[SliceOutput, IsosurfaceOutput],
    entities_name_key: str,
    injection_function,
):
    """Translate slice or isosurface output settings."""
    translated_output = init_output_base(
        output_params, output_class, has_average_capability=False, is_average=False
    )
    shared_output_fields = get_global_setting_from_per_item_setting(
        output_params,
        output_class,
        "output_fields",
        allow_get_from_first_instance_as_fallback=False,
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
        allow_get_from_first_instance_as_fallback=False,
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
        surface_output = translate_surface_output(outputs, SurfaceOutput, translated)
    if has_instance_in_list(outputs, TimeAverageSurfaceOutput):
        surface_output_average = translate_surface_output(
            outputs, TimeAverageSurfaceOutput, translated
        )
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


def porous_media_entity_info_serializer(volume):
    """Porous media entity serializer"""
    if isinstance(volume, Box):
        axes = volume.private_attribute_input_cache.axes
        return {
            "zoneType": "box",
            "axes": [list(axes[0]), list(axes[1])],
            "center": volume.center.value.tolist(),
            "lengths": volume.size.value.tolist(),
        }

    axes = volume.axes
    return {
        "zoneType": "mesh",
        "zoneName": volume.name,
        "axes": [list(axes[0]), list(axes[1])],
    }


def porous_media_translator(model: PorousMedium):
    """Porous media translator"""
    model_dict = remove_units_in_dict(dump_dict(model))
    porous_medium = {
        "DarcyCoefficient": list(model_dict["darcyCoefficient"]),
        "ForchheimerCoefficient": list(model_dict["forchheimerCoefficient"]),
    }
    if model.volumetric_heat_source is not None:
        porous_medium["volumetricHeatSource"] = model_dict["volumetricHeatSource"]

    return porous_medium


def boundary_spec_translator(model: SurfaceModelTypes, op_acousitc_to_static_pressure_ratio):
    """Boundary translator"""
    model_dict = remove_units_in_dict(dump_dict(model))
    boundary = {}
    # pylint: disable=too-many-branches
    if isinstance(model, Wall):
        boundary.update(
            {
                "type": "WallFunction" if model.use_wall_function else "NoSlipWall",
                "velocityType": model.velocity_type,
            }
        )
        if model.velocity:
            boundary["velocity"] = model_dict["velocity"]
        if isinstance(model.heat_spec, Temperature):
            boundary["temperature"] = model_dict["heatSpec"]["value"]
        elif isinstance(model.heat_spec, HeatFlux):
            boundary["heatFlux"] = model_dict["heatSpec"]["value"]
    elif isinstance(model, Inflow):
        boundary["totalTemperatureRatio"] = model_dict["totalTemperature"]
        if model.velocity_direction:
            boundary["velocityDirection"] = model_dict["velocityDirection"]
        if isinstance(model.spec, TotalPressure):
            boundary["type"] = "SubsonicInflow"
            boundary["totalPressureRatio"] = (
                model_dict["spec"]["value"] * op_acousitc_to_static_pressure_ratio
            )
        elif isinstance(model.spec, MassFlowRate):
            boundary["type"] = "MassInflow"
            boundary["massFlowRate"] = model_dict["spec"]["value"]
    elif isinstance(model, Outflow):
        if isinstance(model.spec, Pressure):
            boundary["type"] = "SubsonicOutflowPressure"
            boundary["staticPressureRatio"] = (
                model_dict["spec"]["value"] * op_acousitc_to_static_pressure_ratio
            )
        elif isinstance(model.spec, Mach):
            pass
        elif isinstance(model.spec, MassFlowRate):
            pass
    elif isinstance(model, Periodic):
        pass
    elif isinstance(model, SlipWall):
        boundary["type"] = "SlipWall"
    elif isinstance(model, Freestream):
        boundary.update(
            {
                "type": "Freestream",
                "velocityType": model.velocity_type,
            }
        )

    return boundary


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
    ml = geometry.get("momentLength", 1.0)
    translated["geometry"] = {
        "momentCenter": list(geometry.get("momentCenter", [0.0, 0.0, 0.0])),
        "momentLength": list(ml) if isinstance(ml, tuple) else [ml, ml, ml],
        "refArea": geometry.get("area", 1.0),
    }
    ##:: Step 2: Get freestream
    op = input_params.operating_condition
    translated["freestream"] = {
        "alphaAngle": op.alpha.to("degree").v.item() if "alpha" in op.model_fields else 0,
        "betaAngle": op.beta.to("degree").v.item() if "beta" in op.model_fields else 0,
        "Mach": op.velocity_magnitude.v.item(),
        "Temperature": (
            op.thermal_state.temperature.to("K").v.item()
            if isinstance(op.thermal_state.material.dynamic_viscosity, Sutherland)
            else -1
        ),
        "muRef": op.thermal_state.mu_ref(mesh_unit),
    }
    if "reference_velocity_magnitude" in op.model_fields.keys() and op.reference_velocity_magnitude:
        translated["freestream"]["MachRef"] = op.reference_velocity_magnitude.v.item()
    op_acousitc_to_static_pressure_ratio = (
        op.thermal_state.density * op.thermal_state.speed_of_sound**2 / op.thermal_state.pressure
    ).value

    ##:: Step 3: Get boundaries
    # pylint: disable=duplicate-code
    translated["boundaries"] = translate_setting_and_apply_to_all_entities(
        input_params.models,
        (
            Wall,
            SlipWall,
            Freestream,
            Outflow,
            Inflow,
            Periodic,
            SymmetryPlane,
        ),
        boundary_spec_translator,
        to_list=False,
        op_acousitc_to_static_pressure_ratio=op_acousitc_to_static_pressure_ratio,
    )

    ##:: Step 4: Get outputs (has to be run after the boundaries are translated)

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
    if has_instance_in_list(input_params.models, PorousMedium):
        translated["porousMedia"] = translate_setting_and_apply_to_all_entities(
            input_params.models,
            PorousMedium,
            porous_media_translator,
            to_list=True,
            entity_injection_func=porous_media_entity_info_serializer,
        )

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
                    udd_dict["inputBoundaryPatches"].append(_get_key_name(surface))
            if udd.output_target is not None:
                udd_dict["outputTargetName"] = udd.output_target.name
            translated["userDefinedDynamics"].append(udd_dict)

    return translated
