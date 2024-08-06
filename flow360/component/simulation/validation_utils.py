"""
validation utility functions
"""

from functools import wraps


# pylint: disable=missing-function-docstring
def _get_bet_disk_name(bet_disk):
    disk_name = "one of the BET disks" if bet_disk.name is None else f"BET disk: {bet_disk.name}"
    return disk_name


def _field_validator_append_instance_name(func):
    """
    If the validation throw ValueError (expected), append the instance name to the error message.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the instance name
        prepend_message = None
        model_class = args[0]
        validation_info = args[2]
        if validation_info is not None:
            name = validation_info.data.get("name", None)
            if name is None:
                prepend_message = "In one of the " + model_class.__name__
            else:
                prepend_message = f"{model_class.__name__} with name '{name}'"
        else:
            raise NotImplementedError(
                "[Internal] Make sure your field_validator has validationInfo in the args or"
                " this wrapper is used with a field_validator!!"
            )
        try:
            result = func(*args, **kwargs)  # Call the original function
            return result
        except ValueError as e:
            raise ValueError(f"{prepend_message}: {str(e)}") from e

    return wrapper
