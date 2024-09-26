"""
validation for SimulationParams
"""

from flow360.component.flow360_params.flow360_fields import get_aliases
from flow360.component.simulation.models.solver_numerics import NoneSolver
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput


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


def _check_consistency_ddes_volume_output(v):
    model_type = None
    run_ddes = False

    models = v.models

    if not models:
        return v

    for model in models:
        if isinstance(model, Fluid):
            turbulence_model_solver = model.turbulence_model_solver
            if not isinstance(turbulence_model_solver, NoneSolver) and turbulence_model_solver.DDES:
                model_type = turbulence_model_solver.type_name
                run_ddes = True
                break

    outputs = v.outputs

    if not outputs:
        return v

    for output in outputs:
        if isinstance(output, VolumeOutput) and output.output_fields is not None:
            output_fields = output.output_fields.items
            if "SpalartAllmaras_DDES" in output_fields and not (
                model_type == "SpalartAllmaras" and run_ddes
            ):
                raise ValueError(
                    "SpalartAllmaras_DDES output can only be specified with "
                    "SpalartAllmaras turbulence model and DDES turned on."
                )
            if "kOmegaSST_DDES" in output_fields and not (model_type == "kOmegaSST" and run_ddes):
                raise ValueError(
                    "kOmegaSST_DDES output can only be specified with kOmegaSST turbulence model and DDES turned on."
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
