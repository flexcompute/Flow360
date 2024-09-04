"""
validation for SimulationParams
"""

from flow360.component.flow360_params.flow360_fields import get_aliases
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid
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


def _check_numerical_dissipation_factor_output(v):
    models = v.models

    if not models:
        return v

    low_dissipation_enabled = False

    for model in models:
        if isinstance(model, Fluid) and model.navier_stokes_solver:
            numerical_dissipation_factor = model.navier_stokes_solver.numerical_dissipation_factor
            low_dissipation_flag = int(round(1.0 / numerical_dissipation_factor)) - 1
            if low_dissipation_flag != 0:
                low_dissipation_enabled = True
                break

    if low_dissipation_enabled:
        return v

    outputs = v.outputs

    if not outputs:
        return v

    for output in outputs:
        if not hasattr(output, "output_fields"):
            continue
        if "numericalDissipationFactor" in output.output_fields.items:
            raise ValueError(
                "Numerical dissipation factor output requested, but low dissipation mode is not enabled"
            )

    return v
