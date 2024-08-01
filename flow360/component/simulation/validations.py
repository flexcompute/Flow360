"""
validation logic
"""

from .models.volume_models import BETDisk


def _check_bet_disk(sim_params):
    if sim_params.models is None:
        return sim_params
    bet_disk_index = 0
    for model in sim_params.models:
        if isinstance(model, BETDisk):
            if model.blade_line_chord > 0 and model.initial_blade_direction is None:
                raise ValueError(
                    f"On BET disk index={bet_disk_index}, the initial_blade_direction"
                    "is required to specify since its blade_line_chord is non-zero."
                )
            bet_disk_index += 1
    return sim_params
