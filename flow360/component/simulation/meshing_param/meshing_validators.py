"""Meshing validators — re-import relay."""

from importlib import import_module

_EXPORTED_NAMES = {"validate_snappy_uniform_refinement_entities"}


def __getattr__(name):
    if name not in _EXPORTED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    schema_module = import_module(
        "flow360_schema.models.simulation.meshing_param.meshing_validators"
    )
    value = getattr(schema_module, name)
    globals()[name] = value
    return value
