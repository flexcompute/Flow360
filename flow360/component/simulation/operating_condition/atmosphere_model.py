"""Operating condition atmosphere model — re-import relay."""

from importlib import import_module

_EXPORTED_NAMES = {"StandardAtmosphereModel"}


def __getattr__(name):
    if name not in _EXPORTED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    schema_module = import_module(
        "flow360_schema.models.simulation.operating_condition.atmosphere_model"
    )
    value = getattr(schema_module, name)
    globals()[name] = value
    return value
