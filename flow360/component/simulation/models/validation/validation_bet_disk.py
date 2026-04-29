"""BET disk validation helpers — re-import relay."""

from importlib import import_module


def __getattr__(name):
    schema_module = import_module(
        "flow360_schema.models.simulation.models.validation.validation_bet_disk"
    )
    value = getattr(schema_module, name)
    globals()[name] = value
    return value


def __dir__():
    schema_module = import_module(
        "flow360_schema.models.simulation.models.validation.validation_bet_disk"
    )
    return sorted(set(globals()) | set(vars(schema_module)))
