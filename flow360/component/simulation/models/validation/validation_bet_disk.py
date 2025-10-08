"""
validation BETDisk
"""

from pydantic import ValidationInfo


def _check_bet_disk_initial_blade_direction_and_blade_line_chord(bet_disk):
    if bet_disk.blade_line_chord > 0 and bet_disk.initial_blade_direction is None:
        raise ValueError(
            "the initial_blade_direction"
            " is required to specify since its blade_line_chord is non-zero."
        )
    if bet_disk.initial_blade_direction is not None and bet_disk.blade_line_chord == 0:
        raise ValueError(
            "the blade_line_chord has to be positive"
            " since its initial_blade_direction is specified."
        )
    return bet_disk


# pylint: disable=unused-argument
# This is to enable getting name from the info.
def _check_bet_disk_alphas_in_order(value, info: ValidationInfo):
    if any(value != sorted(value)):
        raise ValueError("the alphas are not in increasing order.")
    return value


def _check_has_duplicate_in_one_radial_list(radial_list):
    existing_radius = set()
    for item in radial_list:
        radius = item.radius.value.item()
        if radius not in existing_radius:
            existing_radius.add(radius)
        else:
            return True, radius
    return False, None


def _check_bet_disk_duplicate_chords(value, info: ValidationInfo):
    has_duplicate, duplicated_radius = _check_has_duplicate_in_one_radial_list(value)
    if has_duplicate:
        raise ValueError(f"it has duplicated radius at {duplicated_radius} in chords.")
    return value


def _check_bet_disk_duplicate_twists(value, info: ValidationInfo):
    has_duplicate, duplicated_radius = _check_has_duplicate_in_one_radial_list(value)
    if has_duplicate:
        raise ValueError(f"it has duplicated radius at {duplicated_radius} in twists.")
    return value


def _check_bet_disk_sectional_radius_and_polars(bet_disk):
    radiuses = bet_disk.sectional_radiuses
    polars = bet_disk.sectional_polars
    if len(radiuses) != len(polars):
        raise ValueError(
            f"the length of sectional_radiuses ({len(radiuses)})"
            f" is not the same as that of sectional_polars ({len(polars)})."
        )
    return bet_disk


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
def _check_3d_coeffs_in_BET_polars(
    coeffs_3d, num_Mach, num_Re, num_alphas, section_index, coeffs_name
):
    if len(coeffs_3d) != num_Mach:
        raise ValueError(
            f"(cross section: {section_index}): number of mach_numbers ="
            f" {num_Mach}, but the first dimension of {coeffs_name} is {len(coeffs_3d)}."
        )
    for index_Mach, coeffs_2d in enumerate(coeffs_3d):
        if len(coeffs_2d) != num_Re:
            raise ValueError(
                f"(cross section: {section_index}) (Mach index (0-based)"
                f" {index_Mach}): number of Reynolds = {num_Re}, "
                f"but the second dimension of {coeffs_name} is {len(coeffs_2d)}."
            )
        for index_Re, coeffs_1d in enumerate(coeffs_2d):
            if len(coeffs_1d) != num_alphas:
                raise ValueError(
                    f"(cross section: {section_index}) "
                    f"(Mach index (0-based) {index_Mach}, Reynolds index (0-based)"
                    f" {index_Re}): number of Alphas = {num_alphas}, "
                    f"but the third dimension of {coeffs_name} is {len(coeffs_1d)}."
                )


def _check_bet_disk_3d_coefficients_in_polars(bet_disk):
    mach_numbers = bet_disk.mach_numbers
    reynolds_numbers = bet_disk.reynolds_numbers
    alphas = bet_disk.alphas
    num_Mach = len(mach_numbers)
    num_Re = len(reynolds_numbers)
    num_alphas = len(alphas)
    polars_all_sections = bet_disk.sectional_polars

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
    return bet_disk
