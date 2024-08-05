"""
validation BETDisk
"""


def _check_bet_disk_initial_blade_direction(bet_disk):
    disk_name = "one BET disk" if bet_disk.name is None else f"the BET disk {bet_disk.name}"
    if bet_disk.blade_line_chord > 0 and bet_disk.initial_blade_direction is None:
        raise ValueError(
            f"On {disk_name}, the initial_blade_direction"
            " is required to specify since its blade_line_chord is non-zero."
        )
    return bet_disk
