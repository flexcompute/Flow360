"""
validation for SimulationParams
"""

from flow360.component.flow360_params.flow360_fields import get_aliases
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.outputs.outputs import SurfaceOutput


def _check_consistency_wall_function_and_surface_output(v):
    has_wall_function_model = False

    models = v.models

    if models is None:
        return v

    for model in models:
        if isinstance(model, Wall) and model.use_wall_function:
            has_wall_function_model = True
            break

    outputs = v.outputs

    if outputs is None:
        return v

    for output in outputs:
        if isinstance(output, SurfaceOutput):
            aliases = get_aliases("wallFunctionMetric", raise_on_not_found=True)
            if [i for i in aliases if i in output.output_fields.items] and (
                not has_wall_function_model
            ):
                raise ValueError(
                    "To use 'wallFunctionMetric' for output specify a Wall with use_wall_function=true"
                )

    return v
