"""
Validation for output parameters
"""

from typing import List, Literal, Union, get_args, get_origin

from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    SurfaceIntegralOutput,
)
from flow360.component.simulation.time_stepping.time_stepping import Steady


def _check_output_fields(params):
    """Check the specified output fields for each output item is valid."""

    # pylint: disable=too-many-branches
    if params.outputs is None:
        return params

    has_legacy_user_defined_field_in_surface_integral_output = False
    for output in params.outputs:
        if isinstance(output, SurfaceIntegralOutput):
            for output_field in output.output_fields.items:
                if isinstance(output_field, str):
                    has_legacy_user_defined_field_in_surface_integral_output = True
                    break
    has_user_defined_fields = len(params.user_defined_fields) > 0

    if (
        has_legacy_user_defined_field_in_surface_integral_output
        and has_user_defined_fields is False
    ):
        raise ValueError(
            "The legacy string output fields in `SurfaceIntegralOutput` must be used with `UserDefinedField`."
        )

    def extract_literal_values(annotation):
        origin = get_origin(annotation)
        if origin is Union:
            # Traverse each Union argument
            results = []
            for arg in get_args(annotation):
                result = extract_literal_values(arg)
                if result:
                    results.extend(result)
            return results
        if origin is list or origin is List:
            # Apply the function to the List's element type
            return extract_literal_values(get_args(annotation)[0])
        if origin is Literal:
            return list(get_args(annotation))
        return []

    additional_fields = [item.name for item in params.user_defined_fields]

    for output_index, output in enumerate(params.outputs):
        if output.output_type in ("AeroAcousticOutput", "StreamlineOutput"):
            continue
        # Get allowed output fields items:
        natively_supported = extract_literal_values(
            output.output_fields.__class__.model_fields["items"].annotation
        )
        allowed_items = natively_supported + additional_fields

        for item in output.output_fields.items:
            if isinstance(item, str) and item not in allowed_items:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}:, {item} is not a"
                    f" valid output field name. Allowed fields are {allowed_items}."
                )

        if output.output_type == "IsosurfaceOutput":
            # using the 1st item's allowed field as all isosurface have same field definition
            allowed_items = (
                extract_literal_values(
                    output.entities.items[0].__class__.model_fields["field"].annotation
                )
                + additional_fields
            )
            for entity in output.entities.items:
                if isinstance(entity.field, str) and entity.field not in allowed_items:
                    raise ValueError(
                        f"In `outputs`[{output_index}] {output.output_type}:, {entity.field} is not a"
                        f" valid iso field name. Allowed fields are {allowed_items}."
                    )

    return params


def _check_output_fields_valid_given_turbulence_model(params):
    """Ensure that the output fields are consistent with the turbulence model used."""

    if not params.models or not params.outputs:
        return params

    turbulence_model = None

    invalid_output_fields = {
        "None": ("kOmega", "nuHat", "residualTurbulence", "solutionTurbulence"),
        "SpalartAllmaras": ("kOmega"),
        "kOmegaSST": ("nuHat"),
    }
    for model in params.models:
        if isinstance(model, Fluid):
            turbulence_model = model.turbulence_model_solver.type_name
            break

    for output_index, output in enumerate(params.outputs):
        if output.output_type in ("AeroAcousticOutput", "StreamlineOutput"):
            continue
        for item in output.output_fields.items:
            if isinstance(item, str) and item in invalid_output_fields[turbulence_model]:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}:, {item} is not a valid"
                    f" output field when using turbulence model: {turbulence_model}."
                )

        if output.output_type == "IsosurfaceOutput":
            for entity in output.entities.items:
                if (
                    isinstance(entity.field, str)
                    and entity.field in invalid_output_fields[turbulence_model]
                ):
                    raise ValueError(
                        f"In `outputs`[{output_index}] {output.output_type}:, {entity.field} is not a valid"
                        f" iso field when using turbulence model: {turbulence_model}."
                    )
    return params


def _check_unsteadiness_to_use_aero_acoustics(params):

    if not params.outputs:
        return params

    if isinstance(params.time_stepping, Steady):

        for output_index, output in enumerate(params.outputs):
            if isinstance(output, AeroAcousticOutput):
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}:"
                    "`AeroAcousticOutput` can only be activated with `Unsteady` simulation."
                )
    # Not running case or is using unsteady
    return params
