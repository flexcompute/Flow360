"""
validation BETDisk
"""


# pylint: disable=missing-function-docstring
def _get_bet_disk_name(bet_disk):
    disk_name = "one BET disk" if bet_disk.name is None else f"the BET disk {bet_disk.name}"
    return disk_name


def _check_bet_disk_initial_blade_direction(bet_disk):
    disk_name = _get_bet_disk_name(bet_disk)
    if bet_disk.blade_line_chord > 0 and bet_disk.initial_blade_direction is None:
        raise ValueError(
            f"On {disk_name}, the initial_blade_direction"
            " is required to specify since its blade_line_chord is non-zero."
        )
    return bet_disk


def _check_bet_disk_alphas_in_order(bet_disk):
    disk_name = _get_bet_disk_name(bet_disk)
    alphas = bet_disk.alphas
    if alphas != sorted(alphas):
        raise ValueError(f"On {disk_name}, the alphas are not in increasing order.")
    return bet_disk


def _check_has_duplicate_in_one_radial_list(radial_list):
    existing_radius = set()
    for item in radial_list:
        radius = item.radius
        if radius not in existing_radius:
            existing_radius.add(radius)
        else:
            return True, radius
    return False, None


def _check_bet_disk_duplicate_chords_or_twists(bet_disk):
    disk_name = _get_bet_disk_name(bet_disk)
    has_duplicate, duplicated_radius = _check_has_duplicate_in_one_radial_list(bet_disk.chords)
    if has_duplicate:
        raise ValueError(
            f"On {disk_name}, it has duplicated radius at {duplicated_radius} in chords."
        )
    has_duplicate, duplicated_radius = _check_has_duplicate_in_one_radial_list(bet_disk.twists)
    if has_duplicate:
        raise ValueError(
            f"On {disk_name}, it has duplicated radius at {duplicated_radius} in twists."
        )
    return bet_disk
