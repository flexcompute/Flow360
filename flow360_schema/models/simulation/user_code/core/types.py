"""Params-dependent helpers for simulation expression workflows."""

from flow360_schema.framework.expression import UserVariable


def get_post_processing_variables(params) -> set[str]:
    """
    Get all the post-processing-related variables from the simulation params.
    """
    post_processing_variables = set()
    for item in params.outputs if params.outputs else []:
        if item.output_type in ("IsosurfaceOutput", "TimeAverageIsosurfaceOutput"):
            for isosurface in item.entities.items:
                if isinstance(isosurface.field, UserVariable):
                    post_processing_variables.add(isosurface.field.name)
        if "output_fields" not in item.__class__.model_fields:
            continue
        for output_field in item.output_fields.items:
            if isinstance(output_field, UserVariable):
                post_processing_variables.add(output_field.name)
    return post_processing_variables
