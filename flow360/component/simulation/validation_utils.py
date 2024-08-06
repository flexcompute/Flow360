"""
validation utility functions
"""


# pylint: disable=missing-function-docstring
def _get_bet_disk_name(bet_disk):
    disk_name = "one of the BET disks" if bet_disk.name is None else f"BET disk: {bet_disk.name}"
    return disk_name
