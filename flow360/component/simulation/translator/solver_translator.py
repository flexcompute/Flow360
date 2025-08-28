"""Flow360 solver setting parameter translator."""

# pylint: disable=too-many-lines
from typing import Type, Union

import unyt as u

from flow360.component.simulation.conversion import (
    LIQUID_IMAGINARY_FREESTREAM_MACH,
    compute_udf_dimensionalization_factor,
)
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.models.material import Sutherland
from flow360.component.simulation.models.solver_numerics import NoneSolver
from flow360.component.simulation.models.surface_models import (
    Freestream,
    HeatFlux,
    Inflow,
    Mach,
    MassFlowRate,
    Outflow,
    Periodic,
    PorousJump,
    Pressure,
    SlaterPorousBleed,
    SlipWall,
    SurfaceModelTypes,
    SymmetryPlane,
    Temperature,
    TotalPressure,
    Translational,
    Wall,
    WallRotation,
    WallVelocityModelTypes,
)
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    AngularVelocity,
    BETDisk,
    Fluid,
    NavierStokesInitialCondition,
    NavierStokesModifiedRestartSolution,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    LiquidOperatingCondition,
)
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
)
from flow360.component.simulation.outputs.output_fields import (
    PREDEFINED_UDF_EXPRESSIONS,
    append_component_to_output_fields,
    generate_predefined_udf,
)
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    ImportedSurfaceIntegralOutput,
    ImportedSurfaceOutput,
    Isosurface,
    IsosurfaceOutput,
    ProbeOutput,
    Slice,
    SliceOutput,
    StreamlineOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
    SurfaceSliceOutput,
    TimeAverageImportedSurfaceOutput,
    TimeAverageIsosurfaceOutput,
    TimeAverageProbeOutput,
    TimeAverageSliceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageSurfaceProbeOutput,
    TimeAverageVolumeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    BOUNDARY_FULL_NAME_WHEN_NOT_FOUND,
    Box,
    ImportedSurface,
    SurfacePair,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.translator.user_expression_utils import (
    udf_prepending_code,
)
from flow360.component.simulation.translator.utils import (
    _get_key_name,
    convert_tuples_to_lists,
    get_global_setting_from_first_instance,
    has_instance_in_list,
    inline_expressions_in_dict,
    preprocess_input,
    remove_units_in_dict,
    replace_dict_key,
    translate_setting_and_apply_to_all_entities,
    translate_value_or_expression_object,
    update_dict_recursively,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    _convert_numeric,
)
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.component.simulation.utils import (
    is_exact_instance,
    is_instance_of_type_in_union,
)
from flow360.exceptions import Flow360TranslationError


def dump_dict(input_params):
    """Dumping param/model to dictionary."""

    result = input_params.model_dump(by_alias=True, exclude_none=True)
    if result.pop("privateAttributeDict", None) is not None:
        result.update(input_params.private_attribute_dict)
    return result


def init_non_average_output(
    base: dict,
    obj_list,
    class_type: Union[SliceOutput, IsosurfaceOutput, VolumeOutput, SurfaceOutput],
):
    """Initialize the common output attribute for non-average output."""
    base["animationFrequency"] = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "frequency",
    )
    base["animationFrequencyOffset"] = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "frequency_offset",
    )
    return base


def init_average_output(
    base: dict,
    obj_list,
    class_type: Union[TimeAverageVolumeOutput, TimeAverageSurfaceOutput],
):
    """Initialize the common output attribute for average output."""
    base["animationFrequencyTimeAverage"] = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "frequency",
    )
    base["animationFrequencyTimeAverageOffset"] = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "frequency_offset",
    )
    base["startAverageIntegrationStep"] = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "start_step",
    )
    return base


def init_output_base(obj_list, class_type: Type, is_average: bool):
    """Initialize the common output attribute."""

    base = {"outputFields": []}
    output_format = get_global_setting_from_first_instance(
        obj_list,
        class_type,
        "output_format",
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
        )
    return base


def add_unused_output_settings_for_comparison(output_dict: dict):
    """
    Add unused output settings for easier debugging/comparisons.
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


def rotation_translator(model: Rotation, params: SimulationParams):
    """Rotation translator"""
    volume_zone = {
        "modelType": "FluidDynamics",
        "isRotatingReferenceFrame": model.rotating_reference_frame_model,
        "referenceFrame": {},
    }
    if model.parent_volume is not None:
        volume_zone["referenceFrame"]["parentVolumeName"] = model.parent_volume.full_name
    spec = dump_dict(model)["spec"]
    if spec is not None:
        spec_value = spec.get("value", None)
        if isinstance(spec_value, str):
            volume_zone["referenceFrame"]["thetaRadians"] = spec_value
        elif (
            spec_value is not None
            and spec_value.get("units", "") == "flow360_angular_velocity_unit"
        ):
            volume_zone["referenceFrame"]["omegaRadians"] = spec_value["value"]
        elif isinstance(model.spec, AngularVelocity):
            volume_zone["referenceFrame"]["omegaRadians"] = translate_value_or_expression_object(
                model.spec.value, params
            )
    return volume_zone


def translate_output_fields(
    output_model: Union[
        SurfaceOutput,
        TimeAverageSurfaceOutput,
        SliceOutput,
        IsosurfaceOutput,
        ProbeOutput,
        SurfaceIntegralOutput,
        SurfaceProbeOutput,
        SurfaceSliceOutput,
        StreamlineOutput,
        ImportedSurfaceOutput,
        TimeAverageImportedSurfaceOutput,
    ],
):
    """Get output fields"""
    output_fields = []

    for item in append_component_to_output_fields(output_model.output_fields.items):
        output_fields.append(item)

    for output_field in output_model.output_fields.items:
        if isinstance(output_field, UserVariable):
            # Remove the UserVariable object and add its name
            output_fields.append(output_field.name)
    # Filter out the UserVariable Dicts
    output_fields = [item for item in output_fields if isinstance(item, str)]
    return {"outputFields": sorted(output_fields)}


def surface_probe_setting_translation_func(entity: Union[SurfaceProbeOutput, SurfaceSliceOutput]):
    """Translate non-entities part of SurfaceProbeOutput"""
    dict_with_merged_output_fields = monitor_translator(entity)
    dict_with_merged_output_fields["surfacePatches"] = [
        surface.full_name
        for surface in entity.target_surfaces.stored_entities
        if surface.full_name != BOUNDARY_FULL_NAME_WHEN_NOT_FOUND
    ]
    return dict_with_merged_output_fields


def monitor_translator(
    output_model: Union[
        ProbeOutput,
        TimeAverageProbeOutput,
        SurfaceProbeOutput,
        TimeAverageSurfaceProbeOutput,
        SurfaceSliceOutput,
    ],
):
    """Monitor translator"""
    monitor_group = translate_output_fields(output_model)
    # Monitor output setting is fine grained to each monitor group. So no need to have different root level keys.
    if not isinstance(output_model, SurfaceSliceOutput):
        monitor_group["computeTimeAverages"] = False
    monitor_group["animationFrequency"] = 1
    monitor_group["animationFrequencyOffset"] = 0
    if isinstance(output_model, (TimeAverageProbeOutput, TimeAverageSurfaceProbeOutput)):
        monitor_group["computeTimeAverages"] = True
        monitor_group["animationFrequencyTimeAverage"] = output_model.frequency
        monitor_group["animationFrequencyTimeAverageOffset"] = output_model.frequency_offset
        monitor_group["startAverageIntegrationStep"] = output_model.start_step
    return monitor_group


def inject_slice_info(entity: Slice):
    """inject entity info"""
    return {
        "sliceOrigin": list(entity.origin.value),
        "sliceNormal": list(entity.normal),
    }


def inject_surface_slice_info(entity: Slice):
    """inject entity info"""
    return {
        "name": entity.name,
        "sliceOrigin": list(entity.origin.value),
        "sliceNormal": list(entity.normal),
    }


def inject_isosurface_info(entity: Isosurface, input_params: SimulationParams):
    """inject entity info"""

    if isinstance(entity.field, UserVariable):
        units = entity.field.value.get_output_units(input_params=input_params)
        surface_field = entity.field.name
        surface_magnitude = (
            entity.iso_value.to(units).v.item()
            if not isinstance(entity.iso_value, float)
            else entity.iso_value
        )
    else:
        surface_field = entity.field
        surface_magnitude = entity.iso_value
    return_dict = {
        "surfaceField": surface_field,
        "surfaceFieldMagnitude": surface_magnitude,
    }

    if entity.wall_distance_clip_threshold is not None:
        return_dict["wallDistanceClipThreshold"] = entity.wall_distance_clip_threshold.v.item()
    return return_dict


def inject_probe_info(entity: EntityList):
    """inject entity info"""

    translated_entity_dict = {
        "start": [],
        "end": [],
        "numberOfPoints": [],
        "type": "lineProbe",
    }
    for item in entity.stored_entities:
        if isinstance(item, PointArray):
            translated_entity_dict["start"].append(item.start.value.tolist())
            translated_entity_dict["end"].append(item.end.value.tolist())
            translated_entity_dict["numberOfPoints"].append(item.number_of_points)
        if isinstance(item, Point):
            translated_entity_dict["start"].append(item.location.value.tolist())
            translated_entity_dict["end"].append(item.location.value.tolist())
            translated_entity_dict["numberOfPoints"].append(1)

    return translated_entity_dict


def inject_surface_probe_info(entity: EntityList):
    """inject entity info"""

    translated_entity_dict = {
        "start": [],
        "end": [],
        "numberOfPoints": [],
        "type": "lineProbe",
    }
    for item in entity.stored_entities:
        if isinstance(item, PointArray):
            translated_entity_dict["start"].append(item.start.value.tolist())
            translated_entity_dict["end"].append(item.end.value.tolist())
            translated_entity_dict["numberOfPoints"].append(item.number_of_points)
        if isinstance(item, Point):
            translated_entity_dict["start"].append(item.location.value.tolist())
            translated_entity_dict["end"].append(item.location.value.tolist())
            translated_entity_dict["numberOfPoints"].append(1)

    return translated_entity_dict


def inject_surface_list_info(entity: EntityList):
    """inject entity info"""
    return {
        "surfaces": [
            surface.full_name
            for surface in entity.stored_entities
            if surface.full_name != BOUNDARY_FULL_NAME_WHEN_NOT_FOUND
        ],
        "type": "surfaceIntegral",
    }


def inject_imported_surface_info(entity: ImportedSurface):
    """inject entity info"""
    return {
        "meshFile": entity.file_name,
    }


def translate_imported_surface_integral_output(
    output_params: list,
):
    """Translate imported surface integral output settings."""
    translated_output = {
        "animationFrequency": -1,
        "animationFrequencyOffset": 0,
    }
    translated_output["surfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        ImportedSurfaceIntegralOutput,
        translation_func=translate_output_fields,
        entity_injection_func=inject_imported_surface_info,
        to_list=False,
    )

    return translated_output


def translate_volume_output(
    output_params: list, volume_output_class: Union[VolumeOutput, TimeAverageVolumeOutput]
):
    """Translate volume output settings."""
    volume_output = init_output_base(
        output_params,
        volume_output_class,
        is_average=volume_output_class is TimeAverageVolumeOutput,
    )
    # Get outputFields
    output_fields = []
    fields = get_global_setting_from_first_instance(
        output_params, volume_output_class, "output_fields"
    )
    output_fields = append_component_to_output_fields(fields.model_dump()["items"])

    for output_field in fields.items:
        if isinstance(output_field, UserVariable):
            output_fields.append(output_field.name)
    # Filter out the UserVariable Dicts
    output_fields = [item for item in output_fields if isinstance(item, str)]
    volume_output.update(
        {
            "outputFields": sorted(output_fields),
        }
    )
    return volume_output


def translate_imported_surface_output(
    output_params: list,
    imported_surface_output_class: Union[ImportedSurfaceOutput, TimeAverageImportedSurfaceOutput],
):
    """Translate imported surface output settings."""

    imported_surface_output = init_output_base(
        output_params,
        imported_surface_output_class,
        is_average=imported_surface_output_class is TimeAverageImportedSurfaceOutput,
    )
    imported_surface_output["surfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        imported_surface_output_class,
        translation_func=translate_output_fields,
        entity_injection_func=inject_imported_surface_info,
        to_list=False,
    )
    return imported_surface_output


def translate_surface_output(
    output_params: list,
    surface_output_class: Union[SurfaceOutput, TimeAverageSurfaceOutput],
    translated: dict,
):
    """Translate surface output settings."""

    assert "boundaries" in translated  #  "Boundaries must be translated before surface output"

    surface_output = init_output_base(
        output_params,
        surface_output_class,
        is_average=surface_output_class is TimeAverageSurfaceOutput,
    )
    surface_output["surfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        surface_output_class,
        translation_func=translate_output_fields,
        to_list=False,
    )
    surface_output["writeSingleFile"] = get_global_setting_from_first_instance(
        output_params,
        surface_output_class,
        "write_single_file",
    )
    return surface_output


def translate_slice_output(
    output_params: list,
    output_class: Union[SliceOutput, TimeAverageSliceOutput],
    injection_function,
):
    """Translate slice or isosurface output settings."""
    translated_output = init_output_base(
        output_params,
        output_class,
        is_average=output_class is TimeAverageSliceOutput,
    )
    translated_output["slices"] = translate_setting_and_apply_to_all_entities(
        output_params,
        output_class,
        translation_func=translate_output_fields,
        to_list=False,
        entity_injection_func=injection_function,
    )
    return translated_output


def translate_isosurface_output(
    input_params: SimulationParams,
    output_class: Union[IsosurfaceOutput, TimeAverageIsosurfaceOutput],
    output_params: list,
    injection_function,
):
    """Translate slice or isosurface output settings."""
    translated_output = init_output_base(
        output_params,
        output_class,
        is_average=output_class is TimeAverageIsosurfaceOutput,
    )
    translated_output["isoSurfaces"] = translate_setting_and_apply_to_all_entities(
        output_params,
        output_class,
        translation_func=translate_output_fields,
        to_list=False,
        entity_injection_func=injection_function,
        entity_injection_input_params=input_params,
    )
    return translated_output


def translate_surface_slice_output(
    output_params: list,
    output_class: Union[SurfaceSliceOutput],
):
    """Translate surface output settings."""

    surface_slice_output = init_output_base(
        output_params,
        output_class,
        is_average=False,
    )
    surface_slice_output["slices"] = translate_setting_and_apply_to_all_entities(
        output_params,
        output_class,
        translation_func=surface_probe_setting_translation_func,
        to_list=True,
        entity_injection_func=inject_surface_slice_info,
    )
    return surface_slice_output


def translate_monitor_output(
    output_params: list,
    monitor_type,
    injection_function,
    translation_func=monitor_translator,
):
    """Translate monitor output settings."""
    translated_output = {"outputFields": []}
    translated_output["monitors"] = translate_setting_and_apply_to_all_entities(
        output_params,
        monitor_type,
        translation_func=translation_func,
        to_list=False,
        entity_injection_func=injection_function,
        lump_list_of_entities=True,
        use_instance_name_as_key=True,
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
            aeroacoustic_output["observers"] = [
                item.position.value.tolist() for item in output.observers
            ]
            aeroacoustic_output["writePerSurfaceOutput"] = output.write_per_surface_output
            aeroacoustic_output["patchType"] = output.patch_type
            if output.permeable_surfaces is not None:
                aeroacoustic_output["permeableSurfaces"] = [
                    item.full_name
                    for item in output.permeable_surfaces.stored_entities
                    if item.full_name != BOUNDARY_FULL_NAME_WHEN_NOT_FOUND
                ]
            return aeroacoustic_output
    return None


def user_variable_to_udf(
    variable: UserVariable, input_params: SimulationParams
):  # pylint:disable=too-many-branches
    # pylint:disable=too-many-statements
    """Convert user variable to UDF"""
    if not isinstance(variable.value, Expression):
        expression = Expression.model_validate(_convert_numeric(variable.value))
    else:
        expression = variable.value

    def _prepare_prepending_code(expression: Expression):
        prepending_code = []
        for name in sorted(expression.solver_variable_names(recursive=True)):
            if not udf_prepending_code.get(name):
                continue
            if name == "solution.temperature" and input_params.has_solid():
                prepending_code.append(udf_prepending_code["solution.temperature_solid"])
                continue
            prepending_code.append(udf_prepending_code[name])
        prepending_code = "".join(prepending_code)
        return prepending_code

    requested_unit: Union[u.Unit, None] = expression.get_output_units(input_params=input_params)
    if requested_unit is None:
        # Number constant output requested
        coefficient = 1
        offset = 0
    else:
        coefficient, offset = compute_udf_dimensionalization_factor(
            params=input_params,
            requested_unit=requested_unit,
            using_liquid_op=isinstance(input_params.operating_condition, LiquidOperatingCondition),
        )

    expression_length = expression.length
    prepending_code = _prepare_prepending_code(expression=expression)
    if expression_length == 0:  # Scalar output requested
        expression = expression.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
        if offset != 0:
            expression = (expression + offset) * coefficient
        else:
            expression = expression * coefficient
        if not isinstance(expression, Expression):
            # Enforce constant as Expression
            expression = Expression.model_validate(_convert_numeric(expression))
        expression = expression.to_solver_code(params=input_params)
        return UserDefinedField(
            name=variable.name, expression=f"{prepending_code}{variable.name} = " + expression + ";"
        )

    # Vector output requested
    expression = [
        expression[i].evaluate(raise_on_non_evaluable=False, force_evaluate=False)
        for i in range(expression_length)
    ]
    if offset != 0:
        expression = [(item + offset) * coefficient for item in expression]
    else:
        expression = [item * coefficient for item in expression]
    for i, item in enumerate(expression):
        if isinstance(item, Expression):
            expression[i] = item.to_solver_code(params=input_params)
        else:
            expression[i] = Expression.model_validate(_convert_numeric(item)).to_solver_code(
                params=input_params
            )

    expression = [f"{variable.name}[{i}] = " + item for i, item in enumerate(expression)]
    return UserDefinedField(
        name=variable.name, expression=prepending_code + "; ".join(expression) + ";"
    )


def process_output_fields_for_udf(input_params: SimulationParams):
    # pylint:disable=too-many-branches
    """
    Process all output fields from different output types and generate additional
    UserDefinedFields for dimensioned fields.

    Args:
        input_params: SimulationParams object containing outputs configuration

    Returns:
        tuple: (all_field_names, generated_udfs) where:
            - all_field_names is a set of all output field names
            - generated_udfs is a list of UserDefinedField objects for dimensioned fields
    """

    # Collect all output field names from all output types
    all_field_names = set()

    if input_params.outputs:
        for output in input_params.outputs:
            if hasattr(output, "output_fields") and output.output_fields:
                all_field_names.update(
                    append_component_to_output_fields(output.output_fields.items)
                )
            if isinstance(output, IsosurfaceOutput):
                for isosurface in output.entities.items:
                    if isosurface.field in PREDEFINED_UDF_EXPRESSIONS:
                        all_field_names.add(isosurface.field)

    if isinstance(input_params.operating_condition, LiquidOperatingCondition):
        all_field_names.add("velocity_magnitude")

    # Generate UDFs for dimensioned fields
    generated_udfs = []
    for field_name in all_field_names:
        udf_expression = generate_predefined_udf(field_name, input_params)
        if udf_expression:
            generated_udfs.append(UserDefinedField(name=field_name, expression=udf_expression))

    # UserVariable handling:
    user_variable_udfs = {}
    if input_params.outputs:
        for output in input_params.outputs:
            if hasattr(output, "output_fields") and output.output_fields:
                for output_field in output.output_fields.items:
                    if not isinstance(output_field, UserVariable):
                        continue
                    udf_from_user_variable = user_variable_to_udf(output_field, input_params)
                    user_variable_udfs[udf_from_user_variable.name] = udf_from_user_variable
            if isinstance(output, IsosurfaceOutput):
                for isosurface in output.entities.items:
                    if not isinstance(isosurface.field, UserVariable):
                        continue
                    udf_from_user_variable = user_variable_to_udf(isosurface.field, input_params)
                    user_variable_udfs[udf_from_user_variable.name] = udf_from_user_variable
    return generated_udfs, list(user_variable_udfs.values())


def translate_streamline_output(output_params: list):
    """Translate streamline output settings."""
    streamline_output = {"Points": [], "PointArrays": [], "PointArrays2D": []}
    for output in output_params:
        if isinstance(output, StreamlineOutput):
            for entity in output.entities.stored_entities:
                if isinstance(entity, Point):
                    point = {"name": entity.name, "location": entity.location.value.tolist()}
                    streamline_output["Points"].append(point)
                elif isinstance(entity, PointArray):
                    line = {
                        "name": entity.name,
                        "start": entity.start.value.tolist(),
                        "end": entity.end.value.tolist(),
                        "numberOfPoints": entity.number_of_points,
                    }
                    streamline_output["PointArrays"].append(line)
                elif isinstance(entity, PointArray2D):
                    parallelogram = {
                        "name": entity.name,
                        "origin": entity.origin.value.tolist(),
                        "uAxisVector": entity.u_axis_vector.value.tolist(),
                        "vAxisVector": entity.v_axis_vector.value.tolist(),
                        "uNumberOfPoints": entity.u_number_of_points,
                        "vNumberOfPoints": entity.v_number_of_points,
                    }
                    streamline_output["PointArrays2D"].append(parallelogram)

    return streamline_output


def process_user_variables_for_integral(
    outputs,
    class_type: Union[SurfaceIntegralOutput, ImportedSurfaceIntegralOutput],
):
    """Multiply UserVariable by area for surface integrals."""
    for output in outputs:
        if not isinstance(output, class_type):
            continue
        output_fields_processed = []
        for output_field in output.output_fields.items:
            if isinstance(output_field, UserVariable):
                expression = Expression.model_validate(output_field.value)
                if expression.length == 0:
                    expression_processed = expression * math.magnitude(solution.node_area_vector)
                else:
                    expression_processed = [
                        expression[i] * math.magnitude(solution.node_area_vector)
                        for i in range(expression.length)
                    ]

                output_fields_processed.append(
                    UserVariable(
                        name=output_field.name + "_integral",
                        value=expression_processed,
                    )
                )
            else:
                output_fields_processed.append(output_field)
        output.output_fields.items = output_fields_processed


def translate_output(input_params: SimulationParams, translated: dict):
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    """Translate output settings."""
    outputs = input_params.outputs

    if outputs is None:
        return translated
    ##:: Step1: Get translated["volumeOutput"/"timeAverageVolumeOutput"]
    volume_output_configs = [
        (VolumeOutput, "volumeOutput"),
        (TimeAverageVolumeOutput, "timeAverageVolumeOutput"),
    ]

    for output_class, output_key in volume_output_configs:
        if has_instance_in_list(outputs, output_class):
            volume_output = translate_volume_output(outputs, output_class)
            if volume_output:
                translated[output_key] = add_unused_output_settings_for_comparison(volume_output)

    ##:: Step2: Get translated["surfaceOutput"]
    surface_output_configs = [
        (SurfaceOutput, "surfaceOutput"),
        (TimeAverageSurfaceOutput, "timeAverageSurfaceOutput"),
    ]
    for output_class, output_key in surface_output_configs:
        if has_instance_in_list(outputs, output_class):
            surface_output = translate_surface_output(outputs, output_class, translated)
            if surface_output:
                translated[output_key] = add_unused_output_settings_for_comparison(surface_output)

    ##:: Step3: Get translated["sliceOutput"]
    slice_output_configs = [
        (SliceOutput, "sliceOutput"),
        (TimeAverageSliceOutput, "timeAverageSliceOutput"),
    ]
    for output_class, output_key in slice_output_configs:
        if has_instance_in_list(outputs, output_class):
            slice_output = translate_slice_output(outputs, output_class, inject_slice_info)
            if slice_output:
                translated[output_key] = add_unused_output_settings_for_comparison(slice_output)

    ##:: Step4: Get translated["isoSurfaceOutput"]
    iso_surface_output_configs = [
        (IsosurfaceOutput, "isoSurfaceOutput"),
        (TimeAverageIsosurfaceOutput, "timeAverageIsoSurfaceOutput"),
    ]
    for output_class, output_key in iso_surface_output_configs:
        if has_instance_in_list(outputs, output_class):
            translated[output_key] = translate_isosurface_output(
                input_params, output_class, outputs, inject_isosurface_info
            )

    ##:: Step5: Get translated["importedSurfaceOutput"]
    if has_instance_in_list(outputs, ImportedSurfaceOutput):
        translated["importedSurfaceOutput"] = translate_imported_surface_output(
            outputs, ImportedSurfaceOutput
        )
    if has_instance_in_list(outputs, TimeAverageImportedSurfaceOutput):
        translated["timeAverageImportedSurfaceOutput"] = translate_imported_surface_output(
            outputs, TimeAverageImportedSurfaceOutput
        )

    ##:: Step6: Get translated["monitorOutput"]
    probe_output = {}
    probe_output_average = {}
    integral_output = {}
    if has_instance_in_list(outputs, ProbeOutput):
        probe_output = translate_monitor_output(outputs, ProbeOutput, inject_probe_info)
    if has_instance_in_list(outputs, TimeAverageProbeOutput):
        probe_output_average = translate_monitor_output(
            outputs, TimeAverageProbeOutput, inject_probe_info
        )

    if has_instance_in_list(outputs, SurfaceIntegralOutput):
        process_user_variables_for_integral(outputs, SurfaceIntegralOutput)
        integral_output = translate_monitor_output(
            outputs, SurfaceIntegralOutput, inject_surface_list_info
        )
    # Merge
    if probe_output or probe_output_average:
        probe_output = merge_monitor_output(probe_output, probe_output_average)
    if probe_output or integral_output:
        translated["monitorOutput"] = merge_monitor_output(probe_output, integral_output)

    ##:: Step6.1: Get translated["surfaceMonitorOutput"]
    surface_monitor_output = {}
    surface_monitor_output_average = {}
    if has_instance_in_list(outputs, SurfaceProbeOutput):
        surface_monitor_output = translate_monitor_output(
            outputs,
            SurfaceProbeOutput,
            inject_surface_probe_info,
            surface_probe_setting_translation_func,
        )
    if has_instance_in_list(outputs, TimeAverageSurfaceProbeOutput):
        surface_monitor_output_average = translate_monitor_output(
            outputs,
            TimeAverageSurfaceProbeOutput,
            inject_surface_probe_info,
            surface_probe_setting_translation_func,
        )
    if surface_monitor_output or surface_monitor_output_average:
        translated["surfaceMonitorOutput"] = merge_monitor_output(
            surface_monitor_output, surface_monitor_output_average
        )

    ##:: Step6: Get translated["surfaceSliceOutput"]
    surface_slice_output = {}
    if has_instance_in_list(outputs, SurfaceSliceOutput):
        surface_slice_output = translate_surface_slice_output(outputs, SurfaceSliceOutput)
        translated["surfaceSliceOutput"] = surface_slice_output

    ##:: Step7: Get translated["aeroacousticOutput"]
    if has_instance_in_list(outputs, AeroAcousticOutput):
        translated["aeroacousticOutput"] = translate_acoustic_output(outputs)

    ##:: Step8: Get translated["streamlineOutput"]
    if has_instance_in_list(outputs, StreamlineOutput):
        translated["streamlineOutput"] = translate_streamline_output(outputs)

    ##:: Step9: Get translated["importedSurfaceIntegralOutput"]
    if has_instance_in_list(outputs, ImportedSurfaceIntegralOutput):
        process_user_variables_for_integral(
            outputs,
            ImportedSurfaceIntegralOutput,
        )
        translated["importedSurfaceIntegralOutput"] = translate_imported_surface_integral_output(
            outputs,
        )

    ##:: Step10: Sort all "output_fields" everywhere
    # Recursively sort all "outputFields" lists in the translated dict
    def _sort_output_fields_in_dict(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "outputFields" and isinstance(v, list):
                    v.sort()
                else:
                    _sort_output_fields_in_dict(v)
        elif isinstance(d, list):
            for item in d:
                _sort_output_fields_in_dict(item)

    _sort_output_fields_in_dict(translated)
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
        "zoneName": volume.full_name,
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


def bet_disk_entity_info_serializer(volume):
    """BET disk entity serializer"""
    v = convert_tuples_to_lists(remove_units_in_dict(dump_dict(volume)))
    return {
        "axisOfRotation": v["axis"],
        "centerOfRotation": v["center"],
        "radius": v["outerRadius"],
        "thickness": v["height"],
    }


def bet_disk_translator(model: BETDisk):
    """BET disk translator"""
    model_dict = convert_tuples_to_lists(remove_units_in_dict(dump_dict(model)))
    model_dict["alphas"] = [alpha.to("degree").value.item() for alpha in model.alphas]
    model_dict["twists"] = [
        {
            "radius": bet_twist.radius.value.item(),
            "twist": bet_twist.twist.to("degree").value.item(),
        }
        for bet_twist in model.twists
    ]
    disk_param = {
        "rotationDirectionRule": model_dict["rotationDirectionRule"],
        "numberOfBlades": model_dict["numberOfBlades"],
        "omega": model_dict["omega"],
        "chordRef": model_dict["chordRef"],
        "nLoadingNodes": model_dict["nLoadingNodes"],
        "bladeLineChord": model_dict["bladeLineChord"],
        "twists": model_dict["twists"],
        "chords": model_dict["chords"],
        "sectionalPolars": model_dict["sectionalPolars"],
        "sectionalRadiuses": model_dict["sectionalRadiuses"],
        "alphas": model_dict["alphas"],
        "MachNumbers": model_dict["machNumbers"],
        "ReynoldsNumbers": model_dict["reynoldsNumbers"],
        "tipGap": model_dict["tipGap"],
    }
    if "initialBladeDirection" in model_dict:
        disk_param["initialBladeDirection"] = model_dict["initialBladeDirection"]
    return disk_param


def actuator_disk_entity_info_serializer(volume):
    """Actuator disk entity serializer"""
    v = convert_tuples_to_lists(remove_units_in_dict(dump_dict(volume)))
    return {
        "axisThrust": v["axis"],
        "center": v["center"],
        "thickness": v["height"],
    }


def actuator_disk_translator(model: ActuatorDisk):
    """Actuator disk translator"""
    return {
        "forcePerArea": convert_tuples_to_lists(
            remove_units_in_dict(dump_dict(model.force_per_area))
        )
    }


def get_solid_zone_boundaries(volume, solid_zone_boundaries: set):
    """Store solid zone boundaries in a set"""
    if volume.private_attribute_zone_boundary_names is not None:
        for boundary in volume.private_attribute_zone_boundary_names.items:
            solid_zone_boundaries.add(boundary)
    else:
        raise Flow360TranslationError(
            f"boundary_name is required but not found in"
            f"`{volume.__name__}` instances in Solid model. \n[For developers]: This error message should not appear."
            "SimulationParams should have caught this!!!",
            input_value=volume,
            location=["models"],
        )

    return {}


def heat_transfer_volume_zone_translator(model: Solid):
    """Heat transfer volume zone translator"""
    model_dict = remove_units_in_dict(dump_dict(model))
    volume_zone = {
        "modelType": "HeatTransfer",
        "thermalConductivity": model_dict["material"]["thermalConductivity"],
    }
    if model.volumetric_heat_source:
        volume_zone["volumetricHeatSource"] = model_dict["volumetricHeatSource"]
    if model.material.specific_heat_capacity and model.material.density:
        volume_zone["heatCapacity"] = (
            model_dict["material"]["density"] * model_dict["material"]["specificHeatCapacity"]
        )
    if model.initial_condition:
        volume_zone["initialCondition"] = {
            "T": model_dict["initialCondition"]["temperature"],
            "T_solid": model_dict["initialCondition"]["temperature"],
        }
    return volume_zone


def boundary_entity_info_serializer(entity, translated_setting, solid_zone_boundaries):
    """Boundary entity info serializer"""
    output = {}
    if isinstance(entity, SurfacePair):
        key1 = _get_key_name(entity.pair[0])
        key2 = _get_key_name(entity.pair[1])
        if BOUNDARY_FULL_NAME_WHEN_NOT_FOUND in (key1, key2):
            return None
        output[key1] = {**translated_setting, "pairedPatchName": key2}
        output[key2] = translated_setting
    else:
        key_name = _get_key_name(entity)
        if key_name == BOUNDARY_FULL_NAME_WHEN_NOT_FOUND:
            return None
        output = {key_name: {**translated_setting}}
        if key_name in solid_zone_boundaries:
            if "temperature" in translated_setting:
                output[key_name]["type"] = "SolidIsothermalWall"
            elif "heatFlux" in translated_setting:
                output[key_name]["type"] = "SolidIsofluxWall"
            else:
                output[key_name]["type"] = "SolidAdiabaticWall"
            if "roughnessHeight" in translated_setting:
                output[key_name].pop("roughnessHeight")
    return output


def _append_turbulence_quantities_to_dict(model, model_dict, boundary):
    """If the boundary model has turbulence quantities, add it to the boundary dict"""
    if model.turbulence_quantities is not None:
        boundary["turbulenceQuantities"] = model_dict["turbulenceQuantities"]
        replace_dict_key(boundary["turbulenceQuantities"], "typeName", "modelType")
    return boundary


def _get_default_mass_outflow_udd(entities, mass_flow_rate):
    """Default mass outflow UDD translator"""
    proportional_coefficient = 1.0e-2
    udd_list = []
    for entity in entities.stored_entities:
        udd = UserDefinedDynamic(
            name=f"massOutflowController_{entity.name}",
            input_vars=["massFlowRate", "area", "hasSupersonicFlow"],
            constants={
                "massFlowRateTarget": mass_flow_rate,
                "Kp": proportional_coefficient,
                "initialStaticPressureRatio": 1.0,
            },
            output_vars={
                "staticPressureRatio": "if (pseudoStep > 0) state[0]; else initialStaticPressureRatio;"
            },
            state_vars_initial_value=["initialStaticPressureRatio"],
            update_law=[
                "if (hasSupersonicFlow and massFlowRate > 0) (1.0 + Kp) * state[0]; else state[0] + "
                + "Kp * (massFlowRate/area - massFlowRateTarget/area);"
            ],
            input_boundary_patches=entities.stored_entities,
            output_target=entity,
        )
        udd_list.append(udd)

    return udd_list


def _get_default_mass_inflow_udd(entities, mass_flow_rate):
    """Default mass inflow UDD translator"""
    proportional_coefficient = 1.0e-2
    udd_list = []
    for entity in entities.stored_entities:
        udd = UserDefinedDynamic(
            name=f"massInflowController_{entity.name}",
            input_vars=["massFlowRate", "area", "hasSupersonicFlow"],
            constants={
                "massFlowRateTarget": mass_flow_rate,
                "Kp": proportional_coefficient,
                "initialTotalPressureRatio": 1.0,
            },
            output_vars={
                "totalPressureRatio": "if (pseudoStep > 0) state[0]; else initialTotalPressureRatio;"
            },
            state_vars_initial_value=["initialTotalPressureRatio"],
            update_law=[
                "if (hasSupersonicFlow and massFlowRate > 0) (1.0 - Kp) * state[0]; else state[0] - "
                + "Kp * (massFlowRate/area - massFlowRateTarget/area);"
            ],
            input_boundary_patches=entities.stored_entities,
            output_target=entity,
        )
        udd_list.append(udd)

    return udd_list


def mass_flow_default_udd(models, user_defined_dynamics):
    """Default mass flow rate translator"""
    mass_flow_user_defined_dynamics = []
    for model in models:
        if isinstance(model, Outflow):
            if isinstance(model.spec, MassFlowRate):
                udd = _get_default_mass_outflow_udd(model.entities, model.spec.value)
                mass_flow_user_defined_dynamics.extend(udd)
        elif isinstance(model, Inflow):
            if isinstance(model.spec, MassFlowRate):
                udd = _get_default_mass_inflow_udd(model.entities, model.spec.value)
                mass_flow_user_defined_dynamics.extend(udd)

    if len(mass_flow_user_defined_dynamics) == 0:
        return user_defined_dynamics

    if user_defined_dynamics is None:
        user_defined_dynamics = []

    user_defined_dynamics.extend(mass_flow_user_defined_dynamics)
    return user_defined_dynamics


# pylint: disable=too-many-branches, too-many-statements
def boundary_spec_translator(model: SurfaceModelTypes, op_acoustic_to_static_pressure_ratio):
    """Boundary translator"""
    model_dict = remove_units_in_dict(dump_dict(model))
    boundary = {}
    if isinstance(model, Wall):
        boundary["type"] = "WallFunction" if model.use_wall_function else "NoSlipWall"
        if model.velocity is not None:
            if not is_instance_of_type_in_union(model.velocity, WallVelocityModelTypes):
                boundary["velocity"] = list(model_dict["velocity"])
            elif isinstance(model.velocity, SlaterPorousBleed):
                boundary["wallVelocityModel"] = {}
                boundary["wallVelocityModel"]["staticPressureRatio"] = (
                    model.velocity.static_pressure.value * op_acoustic_to_static_pressure_ratio
                )
                boundary["wallVelocityModel"]["porosity"] = model.velocity.porosity
                boundary["wallVelocityModel"]["type"] = model.velocity.type_name
                if model.velocity.activation_step is not None:
                    boundary["wallVelocityModel"]["activationStep"] = model.velocity.activation_step
            elif isinstance(model.velocity, WallRotation):
                omega = model.velocity.angular_velocity.value
                axis = model.velocity.axis
                center = model.velocity.center.value
                boundary["velocity"] = [
                    f"{omega * axis[1]} * (z - {center[2]}) - {omega * axis[2]} * (y - {center[1]})",
                    f"{omega * axis[2]} * (x - {center[0]}) - {omega * axis[0]} * (z - {center[2]})",
                    f"{omega * axis[0]} * (y - {center[1]}) - {omega * axis[1]} * (x - {center[0]})",
                ]
            else:
                raise Flow360TranslationError(
                    f"Unsupported wall velocity setting found with type: {type(model.velocity)}",
                    input_value=model,
                    location=["models"],
                )
        if isinstance(model.heat_spec, Temperature):
            boundary["temperature"] = model_dict["heatSpec"]["value"]
        elif isinstance(model.heat_spec, HeatFlux):
            boundary["heatFlux"] = model_dict["heatSpec"]["value"]
        boundary["roughnessHeight"] = model_dict["roughnessHeight"]
        if model.private_attribute_dict is not None:
            boundary = {**boundary, **model.private_attribute_dict}
    elif isinstance(model, Inflow):
        boundary["totalTemperatureRatio"] = model_dict["totalTemperature"]
        if model.velocity_direction is not None:
            boundary["velocityDirection"] = list(model_dict["velocityDirection"])
        if isinstance(model.spec, TotalPressure):
            boundary["type"] = "SubsonicInflow"
            boundary["totalPressureRatio"] = (
                model_dict["spec"]["value"] * op_acoustic_to_static_pressure_ratio
            )
        elif isinstance(model.spec, MassFlowRate):
            boundary["type"] = "MassInflow"
            boundary["massFlowRate"] = model_dict["spec"]["value"]
            if model.spec.ramp_steps is not None:
                boundary["rampSteps"] = model_dict["spec"]["rampSteps"]
        boundary = _append_turbulence_quantities_to_dict(model, model_dict, boundary)
    elif isinstance(model, Outflow):
        if isinstance(model.spec, Pressure):
            boundary["type"] = "SubsonicOutflowPressure"
            boundary["staticPressureRatio"] = (
                model_dict["spec"]["value"] * op_acoustic_to_static_pressure_ratio
            )
        elif isinstance(model.spec, Mach):
            boundary["type"] = "SubsonicOutflowMach"
            boundary["MachNumber"] = model_dict["spec"]["value"]
        elif isinstance(model.spec, MassFlowRate):
            boundary["type"] = "MassOutflow"
            boundary["massFlowRate"] = model_dict["spec"]["value"]
            if model.spec.ramp_steps is not None:
                boundary["rampSteps"] = model_dict["spec"]["rampSteps"]
    elif isinstance(model, Periodic):
        boundary["type"] = (
            "TranslationallyPeriodic"
            if isinstance(model.spec, Translational)
            else "RotationallyPeriodic"
        )
    elif isinstance(model, SlipWall):
        boundary["type"] = "SlipWall"
    elif isinstance(model, Freestream):
        boundary["type"] = "Freestream"
        if model.velocity is not None:
            boundary["velocity"] = list(model_dict["velocity"])
        boundary = _append_turbulence_quantities_to_dict(model, model_dict, boundary)
    elif isinstance(model, SymmetryPlane):
        boundary["type"] = "SymmetryPlane"
    elif isinstance(model, PorousJump):
        boundary["type"] = "PorousJump"
        boundary["DarcyCoefficient"] = model_dict["darcyCoefficient"]
        boundary["ForchheimerCoefficient"] = model_dict["forchheimerCoefficient"]
        boundary["porousJumpThickness"] = model_dict["thickness"]

    return boundary


def get_navier_stokes_initial_condition(
    initial_condition: Union[NavierStokesInitialCondition, NavierStokesModifiedRestartSolution],
):
    """Translate the initial condition for NavierStokes"""
    if is_exact_instance(initial_condition, NavierStokesInitialCondition):
        initial_condition_dict = {
            "type": "initialCondition",
        }
    elif is_exact_instance(initial_condition, NavierStokesModifiedRestartSolution):
        initial_condition_dict = {
            "type": "restartManipulation",
        }
    else:
        raise Flow360TranslationError(
            f"Invalid NavierStokes initial condition type: {type(initial_condition)}",
            input_value=initial_condition,
            location=["models"],
        )
    raw_dict = dump_dict(initial_condition)
    for key in ["rho", "u", "v", "w", "p", "constants"]:
        if key not in raw_dict:  # not set
            continue
        initial_condition_dict[key] = raw_dict[key]
    return initial_condition_dict


def rename_modeling_constants(modeling_constants):
    """Rename the modeling constants to what the solver reads"""
    if modeling_constants.get("typeName", None) == "SpalartAllmarasConsts":
        replace_dict_key(modeling_constants, "CDES", "C_DES")
        replace_dict_key(modeling_constants, "CD", "C_d")
        replace_dict_key(modeling_constants, "CCb1", "C_cb1")
        replace_dict_key(modeling_constants, "CCb2", "C_cb2")
        replace_dict_key(modeling_constants, "CSigma", "C_sigma")
        replace_dict_key(modeling_constants, "CV1", "C_v1")
        replace_dict_key(modeling_constants, "CVonKarman", "C_vonKarman")
        replace_dict_key(modeling_constants, "CW2", "C_w2")
        replace_dict_key(modeling_constants, "CT3", "C_t3")
        replace_dict_key(modeling_constants, "CT4", "C_t4")
        replace_dict_key(modeling_constants, "CMinRd", "C_min_rd")

    if modeling_constants.get("typeName", None) == "kOmegaSSTConsts":
        replace_dict_key(modeling_constants, "CDES1", "C_DES1")
        replace_dict_key(modeling_constants, "CDES2", "C_DES2")
        replace_dict_key(modeling_constants, "CD1", "C_d1")
        replace_dict_key(modeling_constants, "CD2", "C_d2")
        replace_dict_key(modeling_constants, "CSigmaK1", "C_sigma_k1")
        replace_dict_key(modeling_constants, "CSigmaK2", "C_sigma_k2")
        replace_dict_key(modeling_constants, "CSigmaOmega1", "C_sigma_omega1")
        replace_dict_key(modeling_constants, "CSigmaOmega2", "C_sigma_omega2")
        replace_dict_key(modeling_constants, "CAlpha1", "C_alpha1")
        replace_dict_key(modeling_constants, "CBeta1", "C_beta1")
        replace_dict_key(modeling_constants, "CBeta2", "C_beta2")
        replace_dict_key(modeling_constants, "CBetaStar", "C_beta_star")

    modeling_constants.pop("typeName")  # Not read by solver


def update_controls_modeling_constants(controls, translated):
    """Upading the modelingConstants entries for each control"""
    if controls is not None:
        for control in translated["turbulenceModelSolver"]["controls"]:
            control_modeling_constants = control.get("modelingConstants", None)
            if control_modeling_constants is None:
                continue
            rename_modeling_constants(control_modeling_constants)
            control["modelConstants"] = control.pop("modelingConstants")


# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
@preprocess_input
def get_solver_json(
    input_params: SimulationParams,
    # pylint: disable=no-member,unused-argument
    mesh_unit: LengthType.Positive,
):
    """
    Get the solver json from the simulation parameters.
    """

    translated = {}
    ##:: Step 1: Get geometry:
    if input_params.reference_geometry:
        geometry = inline_expressions_in_dict(
            dump_dict(input_params.reference_geometry), input_params
        )
        geometry = remove_units_in_dict(geometry, skip_keys=["private_attribute_area_settings"])
        translated["geometry"] = {}
        if input_params.reference_geometry.area is not None:
            translated["geometry"]["refArea"] = geometry["area"]
        if input_params.reference_geometry.moment_center is not None:
            translated["geometry"]["momentCenter"] = list(geometry["momentCenter"])
        if input_params.reference_geometry.moment_length is not None:
            ml = geometry["momentLength"]
            translated["geometry"]["momentLength"] = (
                list(ml) if isinstance(ml, tuple) else [ml, ml, ml]
            )

    ##:: Step 2: Get freestream
    op = input_params.operating_condition
    # check if all units are flow360:
    _ = remove_units_in_dict(dump_dict(op))
    translated["freestream"] = {
        "alphaAngle": op.alpha.to("degree").v.item() if "alpha" in op.__class__.model_fields else 0,
        "betaAngle": op.beta.to("degree").v.item() if "beta" in op.__class__.model_fields else 0,
        "Mach": translate_value_or_expression_object(op.velocity_magnitude, input_params),
        "Temperature": (
            op.thermal_state.temperature.to("K").v.item()
            if not isinstance(op, LiquidOperatingCondition)
            and isinstance(op.thermal_state.material.dynamic_viscosity, Sutherland)
            else -1
        ),
        "muRef": (
            op.thermal_state.dynamic_viscosity.v.item()
            if not isinstance(op, LiquidOperatingCondition)
            else op.material.dynamic_viscosity.v.item()
        ),
    }
    if (
        "reference_velocity_magnitude" in op.__class__.model_fields.keys()
        and op.reference_velocity_magnitude
    ):
        translated["freestream"]["MachRef"] = op.reference_velocity_magnitude.v.item()
    op_acoustic_to_static_pressure_ratio = (
        (
            op.thermal_state.density
            * op.thermal_state.speed_of_sound**2
            / op.thermal_state.pressure
        ).value
        if not isinstance(op, LiquidOperatingCondition)
        else 1.0
    )

    ##:: Step 5: Get timeStepping
    ts = input_params.time_stepping
    if isinstance(ts, Unsteady):
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": ts.steps,
            "orderOfAccuracy": ts.order_of_accuracy,
            "maxPseudoSteps": ts.max_pseudo_steps,
            "timeStepSize": translate_value_or_expression_object(ts.step_size, input_params),
        }
    elif isinstance(ts, Steady):
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": 1,
            "orderOfAccuracy": 2,  # Solver always want to read this even if it is steady... Setting dummy value here.
            "maxPseudoSteps": ts.max_steps,
            "timeStepSize": "inf",
        }
    dump_dict(input_params.time_stepping)

    ##:: Step 6: Get solver settings and initial condition
    for model in input_params.models:
        if isinstance(model, Fluid):
            if isinstance(op, LiquidOperatingCondition):
                model.navier_stokes_solver.low_mach_preconditioner = True
                model.navier_stokes_solver.low_mach_preconditioner_threshold = (
                    LIQUID_IMAGINARY_FREESTREAM_MACH
                )
            if (
                model.navier_stokes_solver.low_mach_preconditioner
                and model.navier_stokes_solver.low_mach_preconditioner_threshold is None
            ):
                model.navier_stokes_solver.low_mach_preconditioner_threshold = (
                    input_params.operating_condition.mach
                )
            translated["navierStokesSolver"] = dump_dict(model.navier_stokes_solver)

            replace_dict_key(translated["navierStokesSolver"], "typeName", "modelType")
            if isinstance(op, LiquidOperatingCondition) and not (
                model.navier_stokes_solver.private_attribute_dict is not None
                and "typeName" in model.navier_stokes_solver.private_attribute_dict
            ):
                translated["navierStokesSolver"]["modelType"] = "CompressibleIsentropic"
            replace_dict_key(
                translated["navierStokesSolver"],
                "equationEvaluationFrequency",
                "equationEvalFrequency",
            )

            translated["turbulenceModelSolver"] = dump_dict(model.turbulence_model_solver)
            replace_dict_key(
                translated["turbulenceModelSolver"],
                "equationEvaluationFrequency",
                "equationEvalFrequency",
            )
            replace_dict_key(translated["turbulenceModelSolver"], "typeName", "modelType")
            modeling_constants = translated["turbulenceModelSolver"].get("modelingConstants", None)
            if modeling_constants is not None:
                rename_modeling_constants(modeling_constants)
                translated["turbulenceModelSolver"]["modelConstants"] = translated[
                    "turbulenceModelSolver"
                ].pop("modelingConstants")

            if not isinstance(model.turbulence_model_solver, NoneSolver):
                hybrid_model = model.turbulence_model_solver.hybrid_model
                if hybrid_model is not None:
                    if hybrid_model.shielding_function == "DDES":
                        translated["turbulenceModelSolver"]["DDES"] = True
                        translated["turbulenceModelSolver"]["ZDES"] = False
                    if hybrid_model.shielding_function == "ZDES":
                        translated["turbulenceModelSolver"]["ZDES"] = True
                        translated["turbulenceModelSolver"]["DDES"] = False
                    translated["turbulenceModelSolver"][
                        "gridSizeForLES"
                    ] = hybrid_model.grid_size_for_LES
                    translated["turbulenceModelSolver"].pop("hybridModel")
                else:
                    translated["turbulenceModelSolver"]["DDES"] = False
                    translated["turbulenceModelSolver"]["ZDES"] = False
                    translated["turbulenceModelSolver"]["gridSizeForLES"] = "maxEdgeLength"

                update_controls_modeling_constants(
                    model.turbulence_model_solver.controls, translated
                )

            if not isinstance(model.transition_model_solver, NoneSolver):
                # baseline dictionary dump for transition model object
                translated["transitionModelSolver"] = dump_dict(model.transition_model_solver)
                transition_dict = translated["transitionModelSolver"]
                replace_dict_key(transition_dict, "typeName", "modelType")
                replace_dict_key(
                    transition_dict, "equationEvaluationFrequency", "equationEvalFrequency"
                )
                transition_dict.pop("turbulenceIntensityPercent", None)
                replace_dict_key(transition_dict, "NCrit", "Ncrit")

                # build trip region(s) if applicable
                if "tripRegions" in transition_dict:
                    transition_dict["tripRegions"] = []
                    for trip_region in model.transition_model_solver.trip_regions.stored_entities:
                        axes = trip_region.private_attribute_input_cache.axes
                        transition_dict["tripRegions"].append(
                            {
                                "center": list(trip_region.center.value),
                                "size": list(trip_region.size.value),
                                "axes": [list(axes[0]), list(axes[1])],
                            }
                        )

            translated["initialCondition"] = get_navier_stokes_initial_condition(
                model.initial_condition
            )

    ##:: Step 7: Get BET and AD lists
    if has_instance_in_list(input_params.models, BETDisk):
        translated["BETDisks"] = translate_setting_and_apply_to_all_entities(
            input_params.models,
            BETDisk,
            bet_disk_translator,
            to_list=True,
            entity_injection_func=bet_disk_entity_info_serializer,
        )

    if has_instance_in_list(input_params.models, ActuatorDisk):
        translated["actuatorDisks"] = translate_setting_and_apply_to_all_entities(
            input_params.models,
            ActuatorDisk,
            actuator_disk_translator,
            to_list=True,
            entity_injection_func=actuator_disk_entity_info_serializer,
        )

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
                translation_func_params=input_params,
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
    solid_zone_boundaries = set()
    if has_instance_in_list(input_params.models, Solid):
        translated["heatEquationSolver"] = {
            "equationEvalFrequency": get_global_setting_from_first_instance(
                input_params.models,
                Solid,
                ["heat_equation_solver", "equation_evaluation_frequency"],
            ),
            "linearSolver": {
                "maxIterations": get_global_setting_from_first_instance(
                    input_params.models,
                    Solid,
                    ["heat_equation_solver", "linear_solver", "max_iterations"],
                ),
            },
            "absoluteTolerance": get_global_setting_from_first_instance(
                input_params.models,
                Solid,
                ["heat_equation_solver", "absolute_tolerance"],
            ),
            "relativeTolerance": get_global_setting_from_first_instance(
                input_params.models,
                Solid,
                ["heat_equation_solver", "relative_tolerance"],
            ),
            "orderOfAccuracy": get_global_setting_from_first_instance(
                input_params.models,
                Solid,
                ["heat_equation_solver", "order_of_accuracy"],
            ),
            "CFLMultiplier": 1.0,
            "updateJacobianFrequency": 1,
            "maxForceJacUpdatePhysicalSteps": 0,
            "modelType": "HeatEquation",
        }
        linear_solver_absolute_tolerance = get_global_setting_from_first_instance(
            input_params.models,
            Solid,
            ["heat_equation_solver", "linear_solver", "absolute_tolerance"],
        )
        linear_solver_relative_tolerance = get_global_setting_from_first_instance(
            input_params.models,
            Solid,
            ["heat_equation_solver", "linear_solver", "relative_tolerance"],
        )
        if linear_solver_absolute_tolerance:
            translated["heatEquationSolver"]["linearSolver"][
                "absoluteTolerance"
            ] = linear_solver_absolute_tolerance
        if linear_solver_relative_tolerance:
            translated["heatEquationSolver"]["linearSolver"][
                "relativeTolerance"
            ] = linear_solver_relative_tolerance

        volume_zones = translated.get("volumeZones", {})
        volume_zones.update(
            translate_setting_and_apply_to_all_entities(
                input_params.models,
                Solid,
                heat_transfer_volume_zone_translator,
                to_list=False,
                entity_injection_func=get_solid_zone_boundaries,
                entity_injection_solid_zone_boundaries=solid_zone_boundaries,
            )
        )
        translated["volumeZones"] = volume_zones

    ##:: Step 3: Get boundaries (to be run after volume zones are initialized)
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
            PorousJump,
        ),
        boundary_spec_translator,
        to_list=False,
        entity_injection_func=boundary_entity_info_serializer,
        pass_translated_setting_to_entity_injection=True,
        custom_output_dict_entries=True,
        translation_func_op_acoustic_to_static_pressure_ratio=op_acoustic_to_static_pressure_ratio,
        entity_injection_solid_zone_boundaries=solid_zone_boundaries,
    )

    ##:: Step 4: Get outputs (has to be run after the boundaries are translated)

    translated = translate_output(input_params, translated)

    ##:: Step 5: Get user defined fields and auto-generated fields for dimensioned output
    translated["userDefinedFields"] = []
    # Add auto-generated UDFs for dimensioned fields + user variable UDFs
    generated_udfs, user_variable_udfs = process_output_fields_for_udf(input_params)

    # Add user-specified UDFs and auto-generated UDFs for dimensioned fields
    legacy_udf_count = len(input_params.user_defined_fields) + len(generated_udfs)
    for udf_count, udf in enumerate(
        [*input_params.user_defined_fields, *generated_udfs, *user_variable_udfs]
    ):
        udf_dict = {}
        udf_dict["name"] = udf.name
        udf_dict["expression"] = udf.expression
        if udf_count < legacy_udf_count:
            udf_dict["from_user_variables"] = False
        translated["userDefinedFields"].append(udf_dict)

    translated["userDefinedFields"].sort(key=lambda udf: udf["name"])

    ##:: Step 11: Get user defined dynamics
    input_params.user_defined_dynamics = mass_flow_default_udd(
        input_params.models, input_params.user_defined_dynamics
    )

    if input_params.user_defined_dynamics is not None:
        translated["userDefinedDynamics"] = []
        for udd in input_params.user_defined_dynamics:
            udd_dict = dump_dict(udd)
            udd_dict_translated = {}
            udd_dict_translated["dynamicsName"] = udd_dict.pop("name")
            udd_dict_translated["inputVars"] = udd_dict.pop("inputVars", [])
            udd_dict_translated["inputVars"].sort()
            udd_dict_translated["outputVars"] = udd_dict.pop("outputVars", [])
            udd_dict_translated["stateVarsInitialValue"] = udd_dict.pop("stateVarsInitialValue", [])
            udd_dict_translated["updateLaw"] = udd_dict.pop("updateLaw", [])
            udd_dict_translated["constants"] = udd_dict.pop("constants", {})
            if udd.input_boundary_patches is not None:
                udd_dict_translated["inputBoundaryPatches"] = []
                for surface in udd.input_boundary_patches.stored_entities:
                    full_name = _get_key_name(surface)
                    if full_name != BOUNDARY_FULL_NAME_WHEN_NOT_FOUND:
                        udd_dict_translated["inputBoundaryPatches"].append(full_name)
                udd_dict_translated["inputBoundaryPatches"].sort()
            if udd.output_target is not None:
                full_name = udd.output_target.full_name
                if full_name == BOUNDARY_FULL_NAME_WHEN_NOT_FOUND:
                    raise Flow360TranslationError(
                        f"The output target {udd.output_target.name} is not found in the generated volume mesh.",
                        input_value=udd.output_target,
                    )
                udd_dict_translated["outputTargetName"] = full_name
            translated["userDefinedDynamics"].append(udd_dict_translated)

        translated["userDefinedDynamics"].sort(key=lambda udd: udd["dynamicsName"])

    translated["usingLiquidAsMaterial"] = isinstance(
        input_params.operating_condition, LiquidOperatingCondition
    )
    translated["outputRescale"] = {"velocityScale": 1.0}
    if isinstance(input_params.operating_condition, LiquidOperatingCondition):
        translated["outputRescale"]["velocityScale"] = (
            1.0 / translated["freestream"]["MachRef"]
            if "MachRef" in translated["freestream"].keys()
            else 1.0 / translated["freestream"]["Mach"]
        )
    if input_params.private_attribute_dict is not None:
        translated.update(input_params.private_attribute_dict)

    return translated
