"""
validation for SimulationParams
"""

from flow360.component.simulation.models.solver_numerics import NoneSolver
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import (
    Fluid,
    NavierStokesInitialCondition,
    Solid,
)
from flow360.component.simulation.outputs.outputs import (
    IsosurfaceOutput,
    ProbeOutput,
    SliceOutput,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.time_stepping.time_stepping import Unsteady


def _check_consistency_wall_function_and_surface_output(v):
    models = v.models

    if models:
        has_wall_function_model = False
        for model in models:
            if isinstance(model, Wall) and model.use_wall_function:
                has_wall_function_model = True
                break

        if has_wall_function_model:
            return v

    outputs = v.outputs

    if outputs is None:
        return v

    for output in outputs:
        if isinstance(output, SurfaceOutput):
            if "wallFunctionMetric" in output.output_fields.items:
                raise ValueError(
                    "To use 'wallFunctionMetric' for output specify a Wall model with use_wall_function=true. "
                )

    return v


def _check_duplicate_entities_in_models(params):
    models = params.models

    dict_entity = {"Surface": {}, "Volume": {}}

    def register_single_entity(entity, model_type, dict_entity):
        entity_type = None
        if isinstance(entity, _SurfaceEntityBase):
            entity_type = "Surface"
        if isinstance(entity, _VolumeEntityBase):
            entity_type = "Volume"
        entity_log = dict_entity[entity_type].get(entity.name, [])
        entity_log.append(model_type)
        dict_entity[entity_type][entity.name] = entity_log
        return dict_entity

    if models:
        for model in models:
            if hasattr(model, "entities"):
                for entity in model.entities.stored_entities:
                    dict_entity = register_single_entity(entity, model.type, dict_entity)

    error_msg = ""
    for entity_type, entity_model_map in dict_entity.items():
        for entity_name, model_list in entity_model_map.items():
            if len(model_list) > 1:
                model_set = set(model_list)
                model_string = ", ".join(f"`{x}`" for x in sorted(model_set))
                model_string += " models.\n" if len(model_set) > 1 else " model.\n"
                error_msg += (
                    f"{entity_type} entity `{entity_name}` appears "
                    f"multiple times in {model_string}"
                )

    if error_msg != "":
        raise ValueError(error_msg)

    return params


def _check_low_mach_preconditioner_output(v):
    models = v.models

    if models:
        has_low_mach_preconditioner = False
        for model in models:
            if isinstance(model, Fluid) and model.navier_stokes_solver:
                preconditioner = model.navier_stokes_solver.low_mach_preconditioner
                if preconditioner:
                    has_low_mach_preconditioner = True
                    break

        if has_low_mach_preconditioner:
            return v

    outputs = v.outputs

    if not outputs:
        return v

    for output in outputs:
        if not hasattr(output, "output_fields"):
            continue
        if "lowMachPreconditionerSensor" in output.output_fields.items:
            raise ValueError(
                "Low-Mach preconditioner output requested, but low_mach_preconditioner is not enabled. "
                "You can enable it via model.navier_stokes_solver.low_mach_preconditioner = True for a Fluid "
                "model in the models field of the simulation object."
            )

    return v


def _check_numerical_dissipation_factor_output(v):
    models = v.models

    if models:
        low_dissipation_enabled = False
        for model in models:
            if isinstance(model, Fluid) and model.navier_stokes_solver:
                numerical_dissipation_factor = (
                    model.navier_stokes_solver.numerical_dissipation_factor
                )
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
                "Numerical dissipation factor output requested, but low dissipation mode is not enabled. "
                "You can enable it via model.navier_stokes_solver.numerical_dissipation_factor = True for a Fluid "
                "model in the models field of the simulation object."
            )

    return v


def _check_consistency_ddes_volume_output(v):
    model_type = None
    models = v.models

    run_ddes = False

    if models:
        for model in models:
            if isinstance(model, Fluid):
                turbulence_model_solver = model.turbulence_model_solver
                if (
                    not isinstance(turbulence_model_solver, NoneSolver)
                    and turbulence_model_solver.DDES
                ):
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


def _check_cht_solver_settings(params):
    has_heat_transfer = False

    models = params.models

    if models:
        for model in models:
            if isinstance(model, Solid):
                has_heat_transfer = True

        if has_heat_transfer is False:
            params = _validate_cht_no_heat_transfer(params)
        if has_heat_transfer is True:
            params = _validate_cht_has_heat_transfer(params)

    return params


def _validate_cht_no_heat_transfer(params):

    if params.outputs:
        for output in params.outputs:
            if isinstance(
                output, (SurfaceOutput, VolumeOutput, SliceOutput, ProbeOutput, IsosurfaceOutput)
            ):
                if "residualHeatSolver" in output.output_fields.items:
                    raise ValueError(
                        f"Heat equation output variables: residualHeatSolver is requested in {output.output_type} with"
                        " no `Solid` model defined."
                    )

    return params


def _validate_cht_has_heat_transfer(params):

    time_stepping = params.time_stepping
    if isinstance(time_stepping, Unsteady):
        for model_solid in params.models:
            if isinstance(model_solid, Solid):
                if model_solid.material.specific_heat_capacity is None:
                    raise ValueError(
                        "In `Solid` model -> material, the heat capacity needs to be specified "
                        "for unsteady simulations."
                    )
                if model_solid.initial_condition is None:
                    raise ValueError(
                        "In `Solid` model, the initial condition needs to be specified "
                        "for unsteady simulations."
                    )

    for model in params.models:
        if isinstance(model, Fluid) and isinstance(
            model.initial_condition, NavierStokesInitialCondition
        ):
            for model_solid in params.models:
                if isinstance(model_solid, Solid) and model_solid.initial_condition is None:
                    raise ValueError(
                        "In `Solid` model, the initial condition needs to be specified "
                        "when the `Fluid` model uses expression as initial condition."
                    )

    return params
