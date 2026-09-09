"""
validation logic
"""

from copy import deepcopy
from typing import List, Literal, Optional, Tuple, Union, get_args, get_origin

from flow360.component.v1.boundaries import (
    RotationallyPeriodic,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    TranslationallyPeriodic,
    WallFunction,
)
from flow360.component.v1.flow360_fields import _distribute_shared_output_fields
from flow360.component.v1.initial_condition import ExpressionInitialCondition
from flow360.component.v1.params_utils import get_all_output_fields
from flow360.component.v1.solvers import IncompressibleNavierStokesSolver
from flow360.component.v1.time_stepping import SteadyTimeStepping, UnsteadyTimeStepping
from flow360.component.v1.volume_zones import HeatTransferVolumeZone
from flow360.log import log


def _ignore_velocity_type_in_boundaries(values):
    """values here is actually json dict."""
    for obj in values.values():

        if "velocityType" in obj:
            bc_type = obj.get("type", "")
            log.warning(
                f"Specifying velocityType for boundary condition {bc_type} is no longer supported."
                "The boundary velocity type must now always be prescribed relative to to the inertial reference frame."
            )
        if isinstance(obj, dict):
            obj.pop("velocityType", None)
    return values


def _check_tri_quad_boundaries(values):
    boundaries = values.get("boundaries")
    boundary_names = []
    if boundaries is not None:
        boundary_names = list(boundaries.names())
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
        for patch_name in boundaries.names():
            patch_obj = boundaries[patch_name]
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
    has_wall_function_boundary = False
    boundaries = values.get("boundaries")
    if boundaries is not None:
        for boundary_name in boundaries.names():
            if isinstance(boundaries[boundary_name], WallFunction):
                has_wall_function_boundary = True

    surface_output_fields = []
    surface_output = values.get("surface_output")
    if surface_output is not None:
        surface_output_fields = surface_output.output_fields
    fields = surface_output_fields
    if "wallFunctionMetric" in fields and (not has_wall_function_boundary):
        raise ValueError(
            "'wallFunctionMetric' in 'surfaceOutput' is only valid for 'WallFunction' boundary type."
        )
    return values


def _check_consistency_ddes_volume_output(values):
    turbulence_model_solver = values.get("turbulence_model_solver")
    model_type = None
    run_ddes = False
    if turbulence_model_solver is not None:
        model_type = turbulence_model_solver.model_type
        if model_type != "None":
            run_ddes = turbulence_model_solver.DDES

    volume_output = values.get("volumeOutput")
    if volume_output is not None and volume_output.output_fields is not None:
        output_fields = volume_output.output_fields
        if "SpalartAllmaras_DDES" in output_fields and not (
            model_type == "SpalartAllmaras" and run_ddes
        ):
            raise ValueError(
                "SpalartAllmaras_DDES output can only be specified with \
                SpalartAllmaras turbulence model and DDES turned on."
            )
        if "kOmegaSST_DDES" in output_fields and not (model_type == "kOmegaSST" and run_ddes):
            raise ValueError(
                "kOmegaSST_DDES output can only be specified with kOmegaSST turbulence model and DDES turned on."
            )
    return values


# pylint: disable=line-too-long
def _validate_cht_no_heat_transfer_zone(values):
    heat_equation_solver = values.get("heat_equation_solver")
    if heat_equation_solver is not None:
        raise ValueError("Heat equation solver activated with no zone definition.")

    boundaries = values.get("boundaries")
    if boundaries is not None:
        for boundary_name in boundaries.names():
            boundary_prop = boundaries[boundary_name]
            if isinstance(boundary_prop, SolidIsothermalWall):
                raise ValueError(
                    "SolidIsothermalWall boundary is defined with no definition of volume zone of heat transfer."
                )
            if isinstance(boundary_prop, SolidAdiabaticWall):
                raise ValueError(
                    "SolidAdiabaticWall boundary is defined with no definition of volume zone of heat transfer."
                )
    for output_name in [
        "surface_output",
        "volume_output",
        "slice_output",
        "monitor_output",
    ]:
        output_obj = values.get(output_name)
        if output_obj is not None and output_obj.output_fields is not None:
            if "residualHeatSolver" in output_obj.output_fields:
                raise ValueError(
                    "Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer."
                )
    iso_surface_output = values.get("iso_surface_output")
    if iso_surface_output is not None:
        iso_surfaces = iso_surface_output.iso_surfaces
        if iso_surfaces is not None:
            for iso_surface_name in iso_surfaces.names():
                iso_surface_obj = iso_surfaces[iso_surface_name]
                if (
                    iso_surface_obj.output_fields is not None
                    and "residualHeatSolver" in iso_surface_obj.output_fields
                ):
                    raise ValueError(
                        "Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer."
                    )
    return values


def _validate_cht_has_heat_transfer_zone(values):
    navier_stokes_solver = values.get("navier_stokes_solver")
    if navier_stokes_solver is not None and isinstance(
        navier_stokes_solver, IncompressibleNavierStokesSolver
    ):
        raise ValueError("Conjugate heat transfer can not be used with incompressible flow solver.")

    time_stepping = values.get("time_stepping")
    volume_zones = values.get("volume_zones")
    if isinstance(time_stepping, UnsteadyTimeStepping):
        for volume_name in volume_zones.names():
            volume_prop = volume_zones[volume_name]
            if isinstance(volume_prop, HeatTransferVolumeZone):
                if volume_prop.heat_capacity is None:
                    raise ValueError(
                        "Heat capacity needs to be specified for all heat transfer volume zones for unsteady simulations."
                    )
                if volume_prop.initial_condition is None:
                    raise ValueError(
                        "Initial condition needs to be specified for all heat transfer volume zones for unsteady simulations."
                    )

    initial_condition = values.get("initial_condition")
    if isinstance(initial_condition, ExpressionInitialCondition):
        for volume_name in volume_zones.names():
            volume_prop = volume_zones[volume_name]
            if (
                isinstance(volume_prop, HeatTransferVolumeZone)
                and volume_prop.initial_condition is None
            ):
                raise ValueError(
                    "Initial condition needs to be specified for all heat transfer zones for initialization with expressions."
                )
    return values


# pylint: disable=line-too-long
def _check_cht_solver_settings(values):
    has_heat_transfer_zone = False
    volume_zones = values.get("volume_zones")
    if volume_zones is None:
        return values
    for volume_name in volume_zones.names():
        volume_prop = volume_zones[volume_name]
        if isinstance(volume_prop, HeatTransferVolumeZone):
            has_heat_transfer_zone = True
    if has_heat_transfer_zone is False:
        values = _validate_cht_no_heat_transfer_zone(values)
    if has_heat_transfer_zone is True:
        values = _validate_cht_has_heat_transfer_zone(values)

    return values


def _check_eval_frequency_max_pseudo_steps_in_one_solver(max_pseudo_steps, values, solver_name):
    solver = values.get(solver_name)
    solver_eq_eval_freq = None
    if solver is not None and solver.model_type != "None":
        solver_eq_eval_freq = solver.equation_eval_frequency
    if (
        max_pseudo_steps is not None
        and solver_eq_eval_freq is not None
        and max_pseudo_steps < solver_eq_eval_freq
    ):
        raise ValueError(
            f"'equation evaluation frequency' in {solver_name} is greater than max_pseudo_steps."
        )


def _check_equation_eval_frequency_for_unsteady_simulations(values):
    time_stepping = values.get("time_stepping")
    max_pseudo_steps = None
    if isinstance(time_stepping, UnsteadyTimeStepping):
        max_pseudo_steps = time_stepping.max_pseudo_steps

    _check_eval_frequency_max_pseudo_steps_in_one_solver(
        max_pseudo_steps, values, "turbulence_model_solver"
    )
    _check_eval_frequency_max_pseudo_steps_in_one_solver(
        max_pseudo_steps, values, "transition_model_solver"
    )

    return values


def _check_aero_acoustics(values):
    aeroacoustic_output = values.get("aeroacoustic_output")
    boundaries = values.get("boundaries")
    if (
        boundaries is not None
        and aeroacoustic_output is not None
        and len(aeroacoustic_output.observers) > 0
    ):
        for boundary_name in boundaries.names():
            boundary_prop = boundaries[boundary_name]
            if isinstance(boundary_prop, (TranslationallyPeriodic, RotationallyPeriodic, SlipWall)):
                log.warning(
                    f"Aeroacoustic solver is inaccurate for simulations with {boundary_prop.type} boundary condition."
                )
    return values


def _check_incompressible_navier_stokes_solver(values):
    """
    todo: beta feature
    """
    return values


# pylint: disable=line-too-long
def _check_one_periodic_boundary(boundaries, boundary_key, boundary_obj):
    paired_patch_name = boundary_obj.paired_patch_name
    if paired_patch_name is None:
        return
    if paired_patch_name == boundary_key:
        raise ValueError(
            f"{boundary_key}'s paired_patch_name should not be equal to the name of itself."
        )
    if paired_patch_name not in boundaries.names():
        raise ValueError(f"{boundary_key}'s paired_patch_name does not exist in boundaries.")
    paired_patch_obj = boundaries[paired_patch_name]
    if boundary_obj.type != paired_patch_obj.type:
        raise ValueError(
            f"{boundary_key} and its paired boundary {paired_patch_name} do not have the same type of boundary condition."
        )
    if isinstance(boundary_obj, TranslationallyPeriodic):
        if (
            paired_patch_obj.paired_patch_name is not None
            or paired_patch_obj.translation_vector is not None
        ):
            raise ValueError(
                f"Flow360 doesn't allow periodic pairing information of {boundary_key} and {paired_patch_name} specified for both patches."
            )
    elif isinstance(boundary_obj, RotationallyPeriodic):
        if (
            paired_patch_obj.paired_patch_name is not None
            or paired_patch_obj.axis_of_rotation is not None
            or paired_patch_obj.theta_radians is not None
        ):
            raise ValueError(
                f"Flow360 doesn't allow periodic pairing information of {boundary_key} and {paired_patch_name} specified for both patches."
            )


def _check_periodic_boundary_mapping(values):
    boundaries = values.get("boundaries")
    if boundaries is None:
        return values
    for boundary_key in boundaries.names():
        boundary_obj = boundaries[boundary_key]
        if isinstance(boundary_obj, (TranslationallyPeriodic, RotationallyPeriodic)):
            _check_one_periodic_boundary(boundaries, boundary_key, boundary_obj)

    periodic_boundary_keys = set()
    for boundary_key in boundaries.names():
        boundary_obj = boundaries[boundary_key]
        if isinstance(boundary_obj, (TranslationallyPeriodic, RotationallyPeriodic)):
            paired_patch_name = boundary_obj.paired_patch_name
            if paired_patch_name is not None:
                periodic_boundary_keys.add(boundary_key)
                periodic_boundary_keys.add(paired_patch_name)

    for boundary_key in boundaries.names():
        boundary_obj = boundaries[boundary_key]
        if (
            isinstance(boundary_obj, (TranslationallyPeriodic, RotationallyPeriodic))
            and boundary_key not in periodic_boundary_keys
        ):
            raise ValueError(f"Periodic pair for patch {boundary_key} is not specified.")

    return values


def _check_bet_disks_alphas_in_order(disk):
    alphas = disk.get("alphas")
    if alphas != sorted(alphas):
        raise ValueError("alphas are not in increasing order.")
    return disk


# pylint: disable=duplicate-code
def _check_has_duplicate_in_one_radial_list(radial_list) -> Tuple[bool, Optional[float]]:
    existing_radius = set()
    for item in radial_list:
        radius = item.radius
        if radius not in existing_radius:
            existing_radius.add(radius)
        else:
            return True, radius
    return False, None


def _check_bet_disks_duplicate_chords_or_twists(disk):
    chords = disk.get("chords")
    duplicated_radius, has_duplicate = _check_has_duplicate_in_one_radial_list(chords)
    if has_duplicate:
        raise ValueError(f"BET disk has duplicated radius at {duplicated_radius} in chords.")
    twists = disk.get("twists")
    duplicated_radius, has_duplicate = _check_has_duplicate_in_one_radial_list(twists)
    if has_duplicate:
        raise ValueError(f"BET disk has duplicated radius at {duplicated_radius} in twists.")
    return disk


def _check_bet_disks_number_of_defined_polars(disk):
    sectional_radiuses = disk.get("sectional_radiuses")
    sectional_polars = disk.get("sectional_polars")
    if len(sectional_radiuses) != len(sectional_polars):
        raise ValueError(
            f"length of sectional_radiuses ({len(sectional_radiuses)}) is not the same as that of sectional_polars ({len(sectional_polars)})."
        )
    return disk


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
def _check_3d_coeffs_in_BET_polars(
    coeffs_3d, num_Mach, num_Re, num_alphas, section_index, coeffs_name
):
    if len(coeffs_3d) != num_Mach:
        raise ValueError(
            f"(cross section: {section_index}): number of MachNumbers = {num_Mach}, but the first dimension of {coeffs_name} is {len(coeffs_3d)}."
        )
    for index_Mach, coeffs_2d in enumerate(coeffs_3d):
        if len(coeffs_2d) != num_Re:
            raise ValueError(
                f"(cross section: {section_index}) (Mach index (0-based) {index_Mach}): number of Reynolds = {num_Re}, but the second dimension of {coeffs_name} is {len(coeffs_2d)}."
            )
        for index_Re, coeffs_1d in enumerate(coeffs_2d):
            if len(coeffs_1d) != num_alphas:
                raise ValueError(
                    f"(cross section: {section_index}) (Mach index (0-based) {index_Mach}, Reynolds index (0-based) {index_Re}): number of Alphas = {num_alphas}, but the third dimension of {coeffs_name} is {len(coeffs_1d)}."
                )


def _check_bet_disks_3d_coefficients_in_polars(disk):
    mach_numbers = disk.get("mach_numbers")
    reynolds_numbers = disk.get("reynolds_numbers")
    alphas = disk.get("alphas")
    num_Mach = len(mach_numbers)
    num_Re = len(reynolds_numbers)
    num_alphas = len(alphas)
    polars_all_sections = disk.get("sectional_polars")

    for section_index, polars_one_section in enumerate(polars_all_sections):
        lift_coeffs = polars_one_section.lift_coeffs
        drag_coeffs = polars_one_section.drag_coeffs
        if lift_coeffs is not None:
            _check_3d_coeffs_in_BET_polars(
                lift_coeffs,
                num_Mach,
                num_Re,
                num_alphas,
                section_index,
                "lift_coeffs",
            )
        if drag_coeffs is not None:
            _check_3d_coeffs_in_BET_polars(
                drag_coeffs,
                num_Mach,
                num_Re,
                num_alphas,
                section_index,
                "drag_coeffs",
            )
    return disk


def _check_consistency_ddes_unsteady(values):
    time_stepping = values.get("time_stepping")
    turbulence_model_solver = values.get("turbulence_model_solver")
    run_ddes = False
    if turbulence_model_solver is not None and turbulence_model_solver.model_type != "None":
        run_ddes = turbulence_model_solver.DDES
    if run_ddes and (time_stepping is None or isinstance(time_stepping, SteadyTimeStepping)):
        raise ValueError("Running DDES with steady simulation is invalid.")
    return values


def _check_consistency_temperature(values):
    freestream = values.get("freestream")
    fluid_properties = values.get("fluid_properties")

    if (
        freestream is not None
        and fluid_properties is not None
        and hasattr(freestream, "temperature")
    ):
        freestream_temp = freestream.temperature
        fluid_temp = fluid_properties.temperature
        if freestream_temp is not None and fluid_temp is not None and freestream_temp != fluid_temp:
            raise ValueError(
                f"Freestream and fluid property temperature values do not match: "
                f"{freestream_temp} != {fluid_temp}"
            )

    return values


def _get_all_output_fields(values):
    used_output_fields = set()
    used_output_fields.update(get_all_output_fields(values.get("volume_output")))
    used_output_fields.update(get_all_output_fields(values.get("surface_output")))
    used_output_fields.update(get_all_output_fields(values.get("slice_output")))
    used_output_fields.update(get_all_output_fields(values.get("iso_surface_output")))
    used_output_fields.update(get_all_output_fields(values.get("monitor_output")))
    return used_output_fields


def _check_numerical_dissipation_factor_output(values):
    navier_stokes_solver = values.get("navier_stokes_solver")
    if navier_stokes_solver is not None and not isinstance(
        navier_stokes_solver, IncompressibleNavierStokesSolver
    ):
        numerical_dissipation_factor = navier_stokes_solver.numerical_dissipation_factor
        low_dissipation_flag = int(round(1.0 / numerical_dissipation_factor)) - 1
        if low_dissipation_flag == 0 and "numericalDissipationFactor" in _get_all_output_fields(
            values
        ):
            raise ValueError(
                "Numerical dissipation factor output requested, but low dissipation mode is not enabled"
            )
    return values


def _check_low_mach_preconditioner_output(values):
    navier_stokes_solver = values.get("navier_stokes_solver")
    if navier_stokes_solver is not None and not isinstance(
        navier_stokes_solver, IncompressibleNavierStokesSolver
    ):
        low_mach_preconditioner = navier_stokes_solver.low_mach_preconditioner
        if not low_mach_preconditioner and "lowMachPreconditionerSensor" in _get_all_output_fields(
            values
        ):
            raise ValueError(
                "Low-Mach preconditioner output requested, but low-Mach preconditioner mode is not enabled."
            )
    return values


def _check_local_cfl_output(values):
    time_stepping = values.get("time_stepping")
    navier_stokes_solver = values.get("navier_stokes_solver")
    if "localCFL" in _get_all_output_fields(values):
        if time_stepping is None or isinstance(time_stepping, SteadyTimeStepping):
            raise ValueError("Outputting local CFL with steady simulation is invalid.")

        if navier_stokes_solver is not None and isinstance(
            navier_stokes_solver, IncompressibleNavierStokesSolver
        ):
            raise ValueError(
                "Local CFL output requested, but not supported in the incompressible Navier Stokes solver."
            )
    return values


def _check_per_item_output_fields(output_item_obj, additional_fields: List, error_prefix=""):

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

    if output_item_obj.output_fields is not None:
        natively_supported = extract_literal_values(
            output_item_obj.__fields__["output_fields"].annotation
        )
        allowed_items = natively_supported + additional_fields

        for output_field in output_item_obj.output_fields:
            if output_field not in allowed_items:
                raise ValueError(
                    f"{error_prefix}:, {output_field} is not valid output field name. "
                    f"Allowed inputs are {allowed_items}."
                )


def _check_output_fields(values: dict):
    if values.get("user_defined_fields") is not None:
        additional_fields = [item.name for item in values.get("user_defined_fields")]
    else:
        additional_fields = []

    # Volume Output:
    if values.get("volume_output") is not None:
        _check_per_item_output_fields(
            values.get("volume_output"), additional_fields, "volume_output"
        )

    for output_name, collection_name in zip(
        [
            "surface_output",
            "slice_output",
            "iso_surface_output",
            "monitor_output",
        ],
        ["surfaces", "slices", "iso_surfaces", "monitors"],
    ):
        output_obj = values.get(output_name)

        if output_obj is not None:
            output_obj_hardcopy = deepcopy(output_obj)
            collection_obj = getattr(output_obj_hardcopy, collection_name, None)
            if collection_obj is not None:
                # This function modifies the first arg
                _distribute_shared_output_fields(output_obj_hardcopy.__dict__, collection_name)
                for item_name in collection_obj.names():
                    _check_per_item_output_fields(
                        collection_obj[item_name], additional_fields, output_name + "->" + item_name
                    )
            elif (
                getattr(output_obj, "output_fields", None) is not None
            ):  # Did not specify the collection and we add it later:
                _check_per_item_output_fields(output_obj, additional_fields, output_name)
    return values
