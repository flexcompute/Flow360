"""Validation context module — re-import relay."""

from importlib import import_module

_EXPORTED_NAMES = {
    "ALL",
    "CASE",
    "SURFACE_MESH",
    "VOLUME_MESH",
    "CaseField",
    "ConditionalField",
    "ContextField",
    "FeatureUsageInfo",
    "ParamsValidationInfo",
    "TimeSteppingType",
    "ValidationContext",
    "add_validation_warning",
    "context_validator",
    "contextual_field_validator",
    "contextual_model_validator",
    "get_validation_info",
    "get_validation_levels",
    "get_value_with_path",
}


def __getattr__(name):
    if name not in _EXPORTED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    schema_module = import_module("flow360_schema.models.simulation.validation.validation_context")
    value = getattr(schema_module, name)
    globals()[name] = value
    return value
