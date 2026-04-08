"""Output fields — re-import relay."""

from importlib import import_module


def __getattr__(name):
    schema_module = import_module("flow360_schema.models.simulation.outputs.output_fields")
    value = getattr(schema_module, name)
    globals()[name] = value
    return value


def __dir__():
    schema_module = import_module("flow360_schema.models.simulation.outputs.output_fields")
    return sorted(set(globals()) | set(vars(schema_module)))
