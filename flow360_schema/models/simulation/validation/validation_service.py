"""Schema-owned simulation validation and deserialization services."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Literal

import pydantic as pd
from pydantic_core import InitErrorDetails

from flow360_schema.framework.entity.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)
from flow360_schema.framework.expression.registry import clear_context
from flow360_schema.framework.expression.variable import (
    RedeclaringVariableError,
    get_referenced_expressions_and_user_variables,
    restore_variable_space,
)
from flow360_schema.framework.multi_constructor_model_base import parse_model_dict
from flow360_schema.framework.validation.context import DeserializationContext
from flow360_schema.models.entities.volume_entities import Box  # noqa: F401 - required for globals() in parse_model_dict
from flow360_schema import __version__ as _SCHEMA_PACKAGE_VERSION
from flow360_schema.models.simulation.models.volume_models import BETDisk  # noqa: F401
from flow360_schema.models.simulation.operating_condition.operating_condition import (  # noqa: F401
    AerospaceCondition,
    GenericReferenceCondition,
    ThermalState,
)
from flow360_schema.models.simulation.simulation_params import SimulationParams
from flow360_schema.models.simulation.validation.validation_context import (
    ALL,
    ParamsValidationInfo,
    ValidationContext,
)

__all__ = [
    "ALL",
    "ValidationCalledBy",
    "_determine_validation_level",
    "_insert_forward_compatibility_notice",
    "_intersect_validation_levels",
    "_normalize_union_branch_error_location",
    "_populate_error_context",
    "_sanitize_stack_trace",
    "_traverse_error_location",
    "clean_unrelated_setting_from_params_dict",
    "clear_context",
    "handle_generic_exception",
    "initialize_variable_space",
    "validate_error_locations",
    "validate_model",
]


def _intersect_validation_levels(requested_levels, available_levels):
    if requested_levels is not None and available_levels is not None:
        if requested_levels == ALL:
            validation_levels_to_use = [
                item for item in ["SurfaceMesh", "VolumeMesh", "Case"] if item in available_levels
            ]
        elif isinstance(requested_levels, str):
            if requested_levels in available_levels:
                validation_levels_to_use = [requested_levels]
            else:
                validation_levels_to_use = []
        else:
            assert isinstance(requested_levels, list)
            validation_levels_to_use = [item for item in requested_levels if item in available_levels]
        return validation_levels_to_use
    return []


class ValidationCalledBy(Enum):
    """Enum as indicator where ``validate_model()`` is called."""

    LOCAL = "Local"
    SERVICE = "Service"
    PIPELINE = "Pipeline"

    def get_forward_compatibility_error_message(self, version_from: str, version_to: str):
        """Return error message indicating that forward compatibility is limited."""
        error_suffix = " Errors may occur since forward compatibility is limited."
        if self == ValidationCalledBy.LOCAL:
            return {
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "The cloud `SimulationParam` (version: "
                + version_from
                + ") is too new for your local schema package (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        if self == ValidationCalledBy.SERVICE:
            return {
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "Your `SimulationParams` (version: "
                + version_from
                + ") is too new for the solver (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        if self == ValidationCalledBy.PIPELINE:
            return {
                "type": (f"{version_from} > {version_to}"),
                "loc": [],
                "msg": "[Internal] Your `SimulationParams` (version: "
                + version_from
                + ") is too new for the solver (version: "
                + version_to
                + ")."
                + error_suffix,
                "ctx": {},
            }
        return None


def _insert_forward_compatibility_notice(
    validation_errors: list,
    params_as_dict: dict,
    validated_by: ValidationCalledBy,
    version_to: str = _SCHEMA_PACKAGE_VERSION,
):
    version_from = SimulationParams._get_version_from_dict(model_dict=params_as_dict)
    forward_compatibility_failure_error = validated_by.get_forward_compatibility_error_message(
        version_from=version_from,
        version_to=version_to,
    )
    validation_errors.insert(0, forward_compatibility_failure_error)
    return validation_errors


def initialize_variable_space(param_as_dict: dict, use_clear_context: bool = False) -> dict:
    """Load user variables from private attributes into the expression registry."""
    if "private_attribute_asset_cache" not in param_as_dict:
        return param_as_dict

    asset_cache: dict = param_as_dict["private_attribute_asset_cache"]
    if "variable_context" not in asset_cache:
        return param_as_dict
    if not isinstance(asset_cache["variable_context"], Iterable):
        return param_as_dict

    variable_context = asset_cache["variable_context"]

    try:
        restore_variable_space(variable_context, clear_first=use_clear_context)
    except RedeclaringVariableError as error:
        raise ValueError(
            f"Loading user variable '{error.variable_name}' from simulation.json which is "
            "already defined in local context. Please change your local user variable definition."
        ) from error
    except pd.ValidationError as error:
        error_detail: dict = error.errors()[0]
        loc = error_detail.get("loc", ())
        raise pd.ValidationError.from_exception_data(
            "Invalid user variable/expression",
            line_errors=[
                InitErrorDetails(
                    type=error_detail["type"],
                    loc=("private_attribute_asset_cache",) + tuple(loc),
                    input="",
                    ctx=error_detail.get("ctx", {}),
                ),
            ],
        ) from error

    return param_as_dict


def _check_cad_importer_mesher_compatibility(params: SimulationParams) -> dict | None:
    """Reject CAD Importer v2 combined with the standalone beta in-house surface
    mesher (beta enabled without Geometry AI) at the service validation layer.

    The schema model no longer enforces this on deserialize; this guard runs so
    the webservice validate path still rejects the combination before a job
    dispatches. The combination is intrinsically invalid -- it only arises for a
    v2 geometry -- so the check is unconditional (not gated on validation level,
    which is empty when ``root_item_type`` is unknown). Returns a top-level error
    dict (no private-attribute location) or ``None``.
    """
    cache = params.private_attribute_asset_cache
    if cache.cad_importer_version != "v2":
        return None
    if cache.use_inhouse_mesher and not cache.use_geometry_AI:
        return {
            "type": "value_error",
            "loc": (),
            "msg": (
                "The beta in-house surface mesher (without Geometry AI) requires CAD Importer V1. "
                "Re-upload this project with CAD Importer V1, or enable Geometry AI."
            ),
        }
    return None


def validate_model(
    *,
    params_as_dict,
    validated_by: ValidationCalledBy,
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"] | None,
    validation_level: Literal["SurfaceMesh", "VolumeMesh", "Case", "All"] | list | None = ALL,
    version_to: str = _SCHEMA_PACKAGE_VERSION,
) -> tuple[SimulationParams | None, list | None, list[dict[str, Any]]]:
    """Validate a params dictionary against the schema-owned ``SimulationParams`` model."""

    def handle_multi_constructor_model(params_as_dict: dict) -> dict:
        project_length_unit_dict = params_as_dict.get("private_attribute_asset_cache", {}).get(
            "project_length_unit",
            None,
        )
        parse_model_info = ParamsValidationInfo(
            {"private_attribute_asset_cache": {"project_length_unit": project_length_unit_dict}},
            [],
        )
        with (
            ValidationContext(levels=validation_levels_to_use, info=parse_model_info),
            DeserializationContext(),
        ):
            return parse_model_dict(params_as_dict, globals())

    def dict_preprocessing(params_as_dict: dict) -> dict:
        params_as_dict = SimulationParams._sanitize_params_dict(params_as_dict)
        params_as_dict = handle_multi_constructor_model(params_as_dict)
        return materialize_entities_and_selectors_in_place(params_as_dict)

    validation_errors = None
    validation_warnings: list[dict[str, Any]] = []
    validated_param = None
    validation_context: ValidationContext | None = None

    params_as_dict = clean_unrelated_setting_from_params_dict(params_as_dict, root_item_type)
    available_levels = _determine_validation_level(up_to="Case", root_item_type=root_item_type)
    validation_levels_to_use = _intersect_validation_levels(validation_level, available_levels)
    params_as_dict, forward_compatibility_mode = SimulationParams._update_param_dict(
        params_as_dict,
        version_to,
    )

    try:
        updated_param_as_dict = dict_preprocessing(params_as_dict)

        use_clear_context = validated_by == ValidationCalledBy.SERVICE
        initialize_variable_space(updated_param_as_dict, use_clear_context)
        referenced_expressions = get_referenced_expressions_and_user_variables(updated_param_as_dict)

        validation_info = ParamsValidationInfo(
            param_as_dict=updated_param_as_dict,
            referenced_expressions=referenced_expressions,
        )

        with ValidationContext(levels=validation_levels_to_use, info=validation_info) as context:
            validation_context = context
            pre_deserialized_entity_info = validation_info.get_entity_info()
            if pre_deserialized_entity_info is not None:
                updated_param_as_dict = {**updated_param_as_dict}
                updated_param_as_dict["private_attribute_asset_cache"] = {
                    **updated_param_as_dict["private_attribute_asset_cache"],
                    "project_entity_info": pre_deserialized_entity_info,
                }
            with DeserializationContext():
                validated_param = SimulationParams.model_validate(updated_param_as_dict)

    except pd.ValidationError as error:
        validation_errors = error.errors()
    except Exception as error:
        import traceback

        validation_errors = handle_generic_exception(
            error,
            validation_errors,
            loc_prefix=None,
            error_stack=traceback.format_exc(),
        )
    finally:
        if validation_context is not None:
            validation_warnings = list(validation_context.validation_warnings)

    if validation_errors is not None:
        validation_errors = validate_error_locations(validation_errors, params_as_dict)

    if forward_compatibility_mode and validation_errors is not None:
        validation_errors = _insert_forward_compatibility_notice(
            validation_errors,
            params_as_dict,
            validated_by,
            version_to=version_to,
        )

    if validated_param is not None and validation_errors is None:
        mesher_error = _check_cad_importer_mesher_compatibility(validated_param)
        if mesher_error is not None:
            validation_errors = [mesher_error]
            validated_param = None

    return validated_param, validation_errors, validation_warnings


def clean_unrelated_setting_from_params_dict(params: dict, root_item_type: str | None) -> dict:
    """Remove settings that should not participate in the requested validation path."""
    if root_item_type == "VolumeMesh":
        return {k: v for k, v in params.items() if k != "meshing"}
    return params


def _sanitize_stack_trace(stack: str) -> str:
    """Sanitize stack traces to show only ``flow360/`` or ``flow360_schema/`` relative paths."""
    import re

    try:
        stack = re.sub(r"^Traceback \(most recent call last\):\n\s*", "", stack)
        pattern = r'File "[^"]*[/\\]((?:flow360|flow360_schema)[/\\][^"]*)"'
        return re.sub(pattern, r'File "\1"', stack)
    except Exception:
        return stack


def handle_generic_exception(
    err: Exception,
    validation_errors: list | None,
    loc_prefix: list[str] | None = None,
    error_stack: str | None = None,
) -> list:
    """Append a generic exception as a structured validation error."""
    if validation_errors is None:
        validation_errors = []

    error_entry = {
        "type": err.__class__.__name__.lower().replace("error", "_error"),
        "loc": ["unknown"] if loc_prefix is None else loc_prefix,
        "msg": str(err),
        "ctx": {},
    }
    if error_stack is not None:
        error_entry["debug"] = _sanitize_stack_trace(error_stack)

    validation_errors.append(error_entry)
    return validation_errors


def validate_error_locations(errors: list, params: dict) -> list:
    """Normalize and enrich validation error locations against the original params dict."""
    for error in errors:
        current = params
        stripped_positions: set[int] = set()
        for index, field in enumerate(error["loc"][:-1]):
            current, valid = _traverse_error_location(current, field)
            if not valid:
                stripped_positions.add(index)
        if stripped_positions:
            error["loc"] = tuple(item for index, item in enumerate(error["loc"]) if index not in stripped_positions)

        _normalize_union_branch_error_location(error, current)
        _populate_error_context(error)
    return errors


def _normalize_union_branch_error_location(error: dict, current) -> None:
    """Hide internal tagged-union branch names from user-facing error locations."""
    loc = error.get("loc")
    if not isinstance(loc, tuple) or len(loc) == 0:
        return

    branch = loc[-1]
    if branch not in {"number", "expression"}:
        return

    if isinstance(current, dict):
        if branch == "number" and "value" in current:
            error["loc"] = (*loc[:-1], "value")
            return
        if branch == "expression" and "expression" in current:
            error["loc"] = (*loc[:-1], "expression")
            return

    error["loc"] = loc[:-1]


def _traverse_error_location(current, field):
    """Traverse one level of a validation error path against dict/list input."""
    if isinstance(field, int) and isinstance(current, list) and field in range(len(current)):
        return current[field], True
    if isinstance(field, str) and isinstance(current, dict) and field in current:
        return current[field], True
    return current, False


def _populate_error_context(error: dict):
    """Convert error context values to strings for stable serialization."""
    ctx = error.get("ctx")
    if isinstance(ctx, dict):
        for field_name, context in ctx.items():
            try:
                error["ctx"][field_name] = (
                    [str(item) for item in context] if isinstance(context, list) else str(context)
                )
            except Exception:
                error["ctx"][field_name] = "<couldn't stringify>"
    else:
        error["ctx"] = {}


def _determine_validation_level(
    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"],
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"] | None,
) -> list[str] | None:
    if root_item_type is None:
        return None
    all_levels = ["Geometry", "SurfaceMesh", "VolumeMesh", "Case"]
    return all_levels[all_levels.index(root_item_type) + 1 : all_levels.index(up_to) + 1]
