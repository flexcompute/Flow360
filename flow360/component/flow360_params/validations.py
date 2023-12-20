"""
validation logic
"""

from ...log import log
from .boundaries import (
    RotationallyPeriodic,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    TranslationallyPeriodic,
    WallFunction,
)
from .initial_condition import ExpressionInitialCondition
from .time_stepping import UnsteadyTimeStepping
from .volume_zones import HeatTransferVolumeZone


def _check_tri_quad_boundaries(values):
    boundaries = values.get("boundaries")
    boundary_names = []
    if boundaries is not None:
        boundary_names = list(boundaries.get_raw_dict().keys())
    for boundary_name in boundary_names:
        if "/tri_" in boundary_name:
            patch = boundary_name[boundary_name.find("/tri_") + len("/tri_") :]
            quad = boundary_name[0 : boundary_name.find("/tri_")] + "/quad_" + patch
            if quad in boundary_names:
                suggested = boundary_name[0 : boundary_name.find("/tri_") + 1] + patch
                log.warning(
                    f"<{boundary_name}> and <{quad}> found. These may not be valid boundaries and \
                    maybe <{suggested}> should be used instead."
                )
    return values


def _check_duplicate_boundary_name(values):
    boundaries = values.get("boundaries")
    boundary_names = set()
    if boundaries is not None:
        for patch_name, patch_obj in boundaries.get_raw_dict().items():
            if patch_obj.name is not None:
                boundary_name_curr = patch_obj.name
            else:
                boundary_name_curr = patch_name
            if boundary_name_curr in boundary_names:
                raise ValueError(
                    f"Boundary name <{boundary_name_curr}> under patch <{patch_name}> appears multiple times."
                )
            boundary_names.add(boundary_name_curr)
    return values


def _check_consistency_wall_function_and_surface_output(values):
    boundary_types = []
    boundaries = values.get("boundaries")
    if boundaries is not None:
        boundary_types = boundaries.get_subtypes()

    surface_output_fields_root = []
    surface_output = values.get("surfaceOutput")
    if surface_output is not None:
        surface_output_fields_root = surface_output.output_fields
    if "WallFunctionMetric" in surface_output_fields_root and WallFunction not in boundary_types:
        raise ValueError(
            "'WallFunctionMetric' in 'surfaceOutput' is only valid for 'WallFunction' boundary types."
        )
    return values


def _check_consistency_ddes_volume_output(values):
    turbulence_model_solver = values.get("turbulence_model_solver")
    model_type = None
    run_ddes = False
    if turbulence_model_solver is not None:
        model_type = turbulence_model_solver.model_type
        run_ddes = turbulence_model_solver.DDES

    volume_output = values.get("volumeOutput")
    if volume_output is not None and volume_output.output_fields is not None:
        output_fields = volume_output.output_fields
        if "SpalartAllmaras_DDES" in output_fields and not (
            model_type == "SpalartAllmaras" and run_ddes
        ):
            raise ValueError(
                "SpalartAllmaras_DDES output can only be specified with \
                SpalartAllmaras turbulence model and DDES turned on"
            )
        if "kOmegaSST_DDES" in output_fields and not (model_type == "kOmegaSST" and run_ddes):
            raise ValueError(
                "kOmegaSST_DDES output can only be specified with kOmegaSST turbulence model and DDES turned on"
            )
    return values


def _validate_cht_has_heat_transfer_zone(values):
    heat_equation_solver = values.get("heat_equation_solver")
    if heat_equation_solver is not None:
        raise ValueError("Heat equation solver activated with no zone definition")

    boundaries = values.get("boundaries")
    if boundaries is not None:
        for boundary_prop in boundaries.get_raw_dict().values():
            if isinstance(boundary_prop, (SolidIsothermalWall, SolidAdiabaticWall)):
                raise ValueError("CHT boundary defined with no zone definition")
    for output_name in [
        "surface_output",
        "volume_output",
        "slice_output",
        "iso_surface_output",
        "monitor_output",
    ]:
        output_obj = values.get(output_name)
        if output_obj is not None and output_obj.output_fields is not None:
            if "residualHeatSolver" in output_obj.output_fields:
                raise ValueError("Heat equation output variables requested with no zone definition")
    return values


def _validate_cht_no_heat_transfer_zone(values):
    boundaries = values.get("boundaries")
    freestream = values.get("freestream")
    if boundaries is not None and freestream is not None:
        for boundary_prop in boundaries.get_raw_dict().values():
            if isinstance(boundary_prop, SolidIsothermalWall) and freestream.temperature == -1:
                raise ValueError(
                    "Wall temperature is invalid when no freestream reference is specified."
                )
    time_stepping = values.get("time_stepping")
    volume_zones = values.get("volume_zones")
    if time_stepping is not None and isinstance(time_stepping, UnsteadyTimeStepping):
        for volume_prop in volume_zones.get_raw_dict().values():
            if isinstance(volume_prop, HeatTransferVolumeZone):
                if volume_prop.heat_capacity is None:
                    raise ValueError(
                        "Heat capacity needs to be specified for all heat \
                        transfer volume zones for unsteady simulations."
                    )
                if volume_prop.initial_condition is None:
                    raise ValueError(
                        "Initial condition needs to be specified for all \
                        heat transfer volume zones for unsteady simulations."
                    )

    initial_condition = values.get("initial_condition")
    if isinstance(initial_condition, ExpressionInitialCondition):
        for volume_prop in volume_zones.get_raw_dict().values():
            if (
                isinstance(volume_prop, HeatTransferVolumeZone)
                and volume_prop.initial_condition is None
            ):
                raise ValueError(
                    "Initial condition needs to be specified for all \
                    heat transfer zones for initialization with expressions."
                )
    return values


def _check_cht_solver_settings(values):
    has_heat_transfer_zone = False
    volume_zones = values.get("volume_zones")
    if volume_zones is None:
        return values
    for volume_prop in volume_zones.get_raw_dict().values():
        if isinstance(volume_prop, HeatTransferVolumeZone):
            has_heat_transfer_zone = True
    if has_heat_transfer_zone is False:
        values = _validate_cht_has_heat_transfer_zone(values)
    if has_heat_transfer_zone is True:
        values = _validate_cht_no_heat_transfer_zone(values)

    return values


def _check_equation_eval_frequency_for_unsteady_simulations(values):
    time_stepping = values.get("time_stepping")
    turbulence_model_solver = values.get("turbulence_model_solver")
    transition_model_solver = values.get("transition_model_solver")
    max_pseudo_steps = None
    turbulence_eq_eval_freq = None
    transition_eq_eval_freq = None
    if time_stepping is not None and isinstance(time_stepping, UnsteadyTimeStepping):
        max_pseudo_steps = time_stepping.max_pseudo_steps
    if turbulence_model_solver is not None:
        turbulence_eq_eval_freq = turbulence_model_solver.equation_eval_frequency
    if (
        max_pseudo_steps is not None
        and turbulence_eq_eval_freq is not None
        and max_pseudo_steps < turbulence_eq_eval_freq
    ):
        raise ValueError(
            "'equation evaluation frequency' in turbulence_model_solver is greater than max_pseudo_steps."
        )
    if transition_model_solver is not None:
        transition_eq_eval_freq = transition_model_solver.equation_eval_frequency
    if (
        max_pseudo_steps is not None
        and transition_eq_eval_freq is not None
        and max_pseudo_steps < transition_eq_eval_freq
    ):
        raise ValueError(
            "'equation evaluation frequency' in transition_model_solver is greater than max_pseudo_steps."
        )
    return values


def _check_aero_acoustics(values):
    aeroacoustic_output = values.get("aeroacoustic_output")
    boundaries = values.get("boundaries")
    if aeroacoustic_output is not None and len(aeroacoustic_output.observers) > 0:
        for boundary_prop in boundaries.get_raw_dict().values():
            if isinstance(boundary_prop, TranslationallyPeriodic):
                log.warning(
                    "Aeroacoustic solver is inaccurate for simulations with TranslationallyPeriodic boundary condition."
                )
            if isinstance(boundary_prop, RotationallyPeriodic):
                log.warning(
                    "Aeroacoustic solver is inaccurate for simulations with RotationallyPeriodic boundary condition."
                )
            if isinstance(boundary_prop, SlipWall):
                log.warning(
                    "Aeroacoustic solver is inaccurate for simulations with SlipWall boundary condition."
                )
    return values
