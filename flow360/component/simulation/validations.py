"""
validation logic
"""

from .models.volume_models import BETDisk


def _check_bet_disk(sim_params):
    if sim_params.models is None:
        return sim_params
    for model in sim_params.models:
        if isinstance(model, BETDisk):
            disk_name = "one BET disk" if model.name is None else f"the BET disk {model.name}"
            if model.blade_line_chord > 0 and model.initial_blade_direction is None:
                raise ValueError(
                    f"On {disk_name}, the initial_blade_direction"
                    " is required to specify since its blade_line_chord is non-zero."
                )
    return sim_params
