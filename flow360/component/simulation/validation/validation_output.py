"""
Validation for output parameters
"""

from typing import List, Literal, Union, get_args, get_origin


def _check_output_fields(params):
    """Check the specified output fields for each output item is valid."""

    if params.outputs is None:
        return params

    has_surface_integral_output = any(
        output.output_type == "SurfaceIntegralOutput" for output in params.outputs
    )
    has_user_defined_fields = len(params.user_defined_fields) > 0

    if has_surface_integral_output and has_user_defined_fields is False:
        raise ValueError("`SurfaceIntegralOutput` can only be used with `UserDefinedField`.")

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
        if output.output_type == "AeroAcousticOutput":
            continue
        # Get allowed output fields items:
        natively_supported = extract_literal_values(
            output.output_fields.model_fields["items"].annotation
        )
        allowed_items = natively_supported + additional_fields

        for item in output.output_fields.items:
            if item not in allowed_items:
                raise ValueError(
                    f"In `outputs`[{output_index}]:, {item} is not valid output field name. "
                    f"Allowed fields are {allowed_items}."
                )

        if output.output_type == "IsosurfaceOutput":
            # using the 1st item as all items have same field definition as the 1st one.
            allowed_items = (
                extract_literal_values(output.entities.items[0].model_fields["field"].annotation)
                + additional_fields
            )
            for entity in output.entities.items:
                if entity.field not in allowed_items:
                    raise ValueError(
                        f"In `outputs`[{output_index}]:, {entity.field} is not valid iso field name. "
                        f"Allowed fields are {allowed_items}."
                    )

    return params
