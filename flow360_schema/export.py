"""
Export CLI for flow360-schema.

This module provides utilities for exporting Pydantic models to JSON Schema format.

Usage:
    python -m flow360_schema.export Simulation
    python -m flow360_schema.export --list
    python -m flow360_schema.export --all -o dist/schemas/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import unyt as u
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

# JSON Schema Draft-07 URL
JSON_SCHEMA_DRAFT = "http://json-schema.org/draft-07/schema#"

# Field ordering to match frontend convention
FIELD_ORDER = [
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "definitions",
    "properties",
    "items",
    "additionalItems",
    "required",
    "additionalProperties",
    "allOf",
    "anyOf",
    "oneOf",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "enum",
    "const",
    "default",
    "$ref",
    "$units",
    "$version",
    "$displayOrder",
]

_FIELD_ORDER_MAP = {k: i for i, k in enumerate(FIELD_ORDER)}
_DEFAULT_ORDER = len(FIELD_ORDER)


class UnytDefaultToSIJsonSchemaGenerator(GenerateJsonSchema):
    """JSON schema generator that serializes unyt defaults as SI JSON values."""

    def encode_default(self, dft: Any) -> Any:
        """Serialize unyt defaults as SI JSON values instead of dropping them."""
        if isinstance(dft, (u.unyt_array, u.unyt_quantity)):
            return dft.in_base("mks").value.tolist()

        return super().encode_default(dft)


def _hoist_metadata_from_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Hoist custom metadata from anyOf branches to the top level.

    Pydantic generates Optional[Annotated[T, Field(json_schema_extra={"$units": "meter"})]]
    as:
        {
            "anyOf": [
                {"$units": "meter", "type": "number", ...},
                {"type": "null"}
            ]
        }

    common-schema expects $units at the top level:
        {
            "$units": "meter",
            "anyOf": [...]
        }

    This function extracts hoistable keys from the non-null branch.

    Args:
        schema: A schema dict that may contain anyOf.

    Returns:
        Schema with metadata hoisted to top level.
    """
    if "anyOf" not in schema:
        return schema

    # Find non-null branches
    non_null_branches = [branch for branch in schema["anyOf"] if branch.get("type") != "null"]

    # Keys that should be hoisted from branches to top level
    hoistable_keys = ["$units"]

    if len(non_null_branches) != 1:
        # Multiple non-null branches - collect common metadata across all
        # Only hoist if ALL non-null branches have the key AND the same value
        result = dict(schema)

        for key in hoistable_keys:
            values = [branch.get(key) for branch in non_null_branches if key in branch]
            # Ensure ALL branches have this key (not just some)
            if (
                len(values) == len(non_null_branches)
                and values
                and all(v == values[0] for v in values)
                and key not in result
            ):
                result[key] = values[0]

        return result

    # Single non-null branch - hoist its metadata
    branch = non_null_branches[0]
    result = dict(schema)
    for key in hoistable_keys:
        if key in branch and key not in result:
            result[key] = branch[key]

    return result


def _replace_special_floats(value: Any) -> Any:
    """Replace infinity and NaN floats with null for JSON compatibility.

    flow360-schema does not allow infinity or NaN values. Model owners should
    fix their Pydantic models to avoid these values. This function converts
    them to null as a fallback.
    """
    if isinstance(value, dict):
        return {k: _replace_special_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_special_floats(v) for v in value]
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def _inline_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all local $ref references, removing definitions/$defs.

    PropertyCompositionSchema doesn't allow definitions - all references must be
    inlined. This function resolves local refs (#/definitions/... or #/$defs/...)
    and removes the definitions block.

    Args:
        schema: JSON Schema with potential definitions.

    Returns:
        Schema with all local refs inlined and definitions removed.
    """
    # Extract definitions (could be $defs or definitions)
    definitions = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    if not definitions:
        return schema

    def resolve_ref(obj: Any, depth: int = 0) -> Any:
        """Recursively resolve $ref references."""
        # PropertyCompositionSchema only allows 2 levels of reference depth
        # Hitting this limit indicates circular references
        if depth > 10:
            raise ValueError(
                "Maximum reference resolution depth exceeded while inlining local "
                "$ref definitions. This likely indicates a circular reference in "
                "the schema, which is not allowed in PropertyCompositionSchema."
            )

        if isinstance(obj, dict):
            # Check for local $ref
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/definitions/") or ref.startswith("#/$defs/"):
                    # Extract definition name
                    def_name = ref.split("/")[-1]
                    if def_name in definitions:
                        # Get the definition and resolve any nested refs
                        resolved = resolve_ref(definitions[def_name].copy(), depth + 1)
                        # Merge any additional properties from the $ref object
                        # (like title, description, $units) - they override definition
                        for key, value in obj.items():
                            if key != "$ref":
                                resolved[key] = value
                        return resolved
                # External $ref - keep as-is
                return obj
            # Recurse into dict values
            return {k: resolve_ref(v, depth) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_ref(item, depth) for item in obj]
        return obj

    # Resolve all refs
    result = resolve_ref(schema)

    # Remove definitions from result
    result.pop("definitions", None)
    result.pop("$defs", None)

    return cast(dict[str, Any], result)


def _order_schema_fields(schema: Any, _inside_properties: bool = False) -> Any:
    """Reorder schema fields to match frontend convention.

    Args:
        schema: The schema or value to process.
        _inside_properties: If True, we're inside a "properties" dict and should
                           not reorder keys (they are user-defined property names).
    """
    if isinstance(schema, dict):
        if _inside_properties:
            # Don't reorder user-defined property names, just recurse into values
            return {k: _order_schema_fields(v, _inside_properties=False) for k, v in schema.items()}
        # Sort schema keywords, but mark "properties" values as user-defined
        return {
            k: _order_schema_fields(v, _inside_properties=(k == "properties"))
            for k, v in sorted(schema.items(), key=lambda x: _FIELD_ORDER_MAP.get(x[0], _DEFAULT_ORDER))
        }
    if isinstance(schema, list):
        return [_order_schema_fields(item, _inside_properties=False) for item in schema]
    return schema


def _order_top_level_schema_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Reorder only the top-level schema keywords without touching nested content."""
    return dict(sorted(schema.items(), key=lambda item: _FIELD_ORDER_MAP.get(item[0], _DEFAULT_ORDER)))


def _generate_display_order(schema: dict[str, Any]) -> list[str]:
    """Generate default field display order from exported property names."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []

    return [
        key for key in properties if not key.startswith("private_attribute") and not key.startswith("privateAttribute")
    ]


def normalize_schema(
    schema: dict[str, Any],
    _is_root: bool = True,
    inline_defs: bool = False,
) -> dict[str, Any]:
    """
    Normalize JSON Schema for common-schema compatibility.

    Pydantic 2.x generates JSON Schema 2020-12, but common-schema expects Draft-07.
    This function handles the key differences:

    - prefixItems (2020-12) → items as array (Draft-07) for tuples
    - items with prefixItems → additionalItems (Draft-07)
    - $defs (2020-12) → definitions (Draft-07)
    - Removes pydantic-specific fields (validate_default, json_schema_extra)
    - Converts ge/le/gt/lt to minimum/maximum/exclusiveMinimum/exclusiveMaximum
    - Adds additionalProperties: false for objects
    - Converts infinity to "Infinity" string
    - Adds $schema draft-07 if missing (root only)
    - Reorders fields to match frontend convention
    - Hoists $units and other metadata from anyOf branches to top level

    Args:
        schema: JSON Schema to normalize.
        _is_root: Internal flag, True for root schema.
        inline_defs: If True, inline all definitions. If False (default), keep definitions.

    Returns:
        Normalized JSON Schema compatible with common-schema.
    """
    if not isinstance(schema, dict):
        # Handle special float values (infinity, NaN) - convert to null
        if isinstance(schema, float) and (math.isinf(schema) or math.isnan(schema)):
            return None
        return schema

    result: dict[str, Any] = {}

    for key, value in schema.items():
        # Skip fields not part of common-schema's allowed key set.
        # - validate_default, json_schema_extra: pydantic-internal.
        # - conditionally_required, relevant_for: Flow360 runtime metadata injected via
        #   Field(json_schema_extra={...}); consumed by validators in base_model.py, not by schema consumers.
        # - discriminator: OpenAPI extension Pydantic emits for discriminated unions; not standard JSON Schema.
        # - deprecated: standard JSON Schema keyword but not in common-schema's allowed set today.
        # - strictType: Flow360 metadata set by geometric_types' __get_pydantic_json_schema__;
        #   describes a vector shape hint for runtime use, not a JSON Schema keyword.
        # - propertyNames: standard JSON Schema keyword for constraining object key names;
        #   not in common-schema's allowed set today (semantic loss — track for restoration).
        if key in (
            "validate_default",
            "json_schema_extra",
            "conditionally_required",
            "relevant_for",
            "discriminator",
            "deprecated",
            "strictType",
            "propertyNames",
        ):
            continue

        if key == "prefixItems":
            # Convert prefixItems to items array (Draft-07 tuple format)
            result["items"] = [normalize_schema(item, _is_root=False) for item in value]
        elif key == "items" and "prefixItems" in schema:
            # When both prefixItems and items exist in 2020-12,
            # items becomes additionalItems in Draft-07
            if isinstance(value, bool):
                result["additionalItems"] = value
            else:
                result["additionalItems"] = normalize_schema(value, _is_root=False)
        elif key == "$defs":
            # Convert $defs to definitions
            result["definitions"] = {k: normalize_schema(v, _is_root=False) for k, v in value.items()}
        elif key == "$ref" and isinstance(value, str):
            # Update $ref paths from $defs to definitions
            result[key] = value.replace("#/$defs/", "#/definitions/")
        elif key == "items" and "prefixItems" not in schema:
            # Regular items (not a tuple), recurse
            if isinstance(value, list):
                result[key] = [normalize_schema(item, _is_root=False) for item in value]
            elif isinstance(value, dict):
                result[key] = normalize_schema(value, _is_root=False)
            else:
                result[key] = value
        elif key == "properties":
            # Recurse into properties
            result[key] = {k: normalize_schema(v, _is_root=False) for k, v in value.items()}
        elif key in ("allOf", "anyOf", "oneOf"):
            # Recurse into composition operators
            result[key] = [normalize_schema(item, _is_root=False) for item in value]
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = normalize_schema(value, _is_root=False)
        elif key == "additionalItems" and isinstance(value, dict):
            # Normalize additionalItems schema (when already present, not from conversion)
            result[key] = normalize_schema(value, _is_root=False)
        elif key == "definitions":
            # Already Draft-07, but recurse
            result[key] = {k: normalize_schema(v, _is_root=False) for k, v in value.items()}
        # Convert pydantic bounds to JSON Schema bounds
        elif key == "ge" and "minimum" not in schema:
            result["minimum"] = value
        elif key == "le" and "maximum" not in schema:
            result["maximum"] = value
        elif key == "gt" and "exclusiveMinimum" not in schema:
            result["exclusiveMinimum"] = value
        elif key == "lt" and "exclusiveMaximum" not in schema:
            result["exclusiveMaximum"] = value
        elif key in ("ge", "le", "gt", "lt"):
            # Skip if already have the JSON Schema equivalent
            continue
        else:
            # Copy other keys as-is
            result[key] = value

    # Ensure additionalProperties: false for objects (required by PropertyCompositionSchema)
    # Must be exactly false - not True, not a dict schema, not missing
    if (result.get("type") == "object" or "properties" in result) and result.get("additionalProperties") is not False:
        result["additionalProperties"] = False

    # Handle special float values (infinity, NaN) in default/const/enum values
    for field in ("default", "const", "enum"):
        if field in result:
            result[field] = _replace_special_floats(result[field])

    # Add $schema for root schema if missing
    if _is_root and "$schema" not in result:
        result["$schema"] = JSON_SCHEMA_DRAFT

    # For root schema, optionally inline definitions
    if _is_root and inline_defs:
        result = _inline_definitions(result)

    # Hoist metadata from anyOf branches to top level
    result = _hoist_metadata_from_anyof(result)

    # Reorder fields to match frontend convention (after inlining to preserve order)
    result = _order_schema_fields(result)

    return result


def get_exportable_models() -> dict[str, type[BaseModel]]:
    """
    Return a dictionary of all exportable Pydantic models.

    Returns:
        Dict mapping model names to model classes.
    """
    # Import models dynamically to avoid circular imports
    models: dict[str, type[BaseModel]] = {}

    try:
        from flow360_schema import models as models_module

        # Import the base class to exclude it from export
        from flow360_schema.framework.base_model import Flow360BaseModel
        from flow360_schema.models.simulation.simulation_params import SimulationParams

        for name in dir(models_module):
            obj = getattr(models_module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj is not Flow360BaseModel  # Skip abstract base class
                and not name.startswith("_")
            ):
                models[name] = obj

        models["SimulationParams"] = SimulationParams
    except ImportError:
        pass

    return models


def export_schema(
    model_class: type[BaseModel],
    title: str | None = None,
    normalize: bool = True,
    inline_defs: bool = False,
) -> dict[str, Any]:
    """
    Export a Pydantic model to JSON Schema.

    Args:
        model_class: The Pydantic model class to export.
        title: Optional title override for the schema.
        normalize: If True (default), normalize schema for common-schema compatibility.
                   Set to False to keep Pydantic's native output format.
        inline_defs: If True, inline all definitions. If False (default), keep definitions block.

    Returns:
        JSON Schema as a dictionary.
    """
    schema = model_class.model_json_schema(
        mode="validation",
        schema_generator=UnytDefaultToSIJsonSchemaGenerator,
    )

    # Override title if provided (model_json_schema doesn't accept title directly)
    if title:
        schema["title"] = title

    if normalize:
        schema = normalize_schema(schema, inline_defs=inline_defs)
        if "properties" in schema and "$displayOrder" not in schema:
            schema["$displayOrder"] = _generate_display_order(schema)
            schema = _order_top_level_schema_fields(schema)

    return schema


def export_all_schemas(
    output_dir: str | Path,
    normalize: bool = True,
    inline_defs: bool = False,
) -> list[Path]:
    """
    Export all available schemas to a directory.

    Args:
        output_dir: Directory to write schema files to.
        normalize: If True (default), normalize schemas for common-schema compatibility.
        inline_defs: If True, inline all definitions. If False (default), keep definitions block.

    Returns:
        List of paths to exported schema files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = get_exportable_models()
    exported_files: list[Path] = []

    for name, model_class in models.items():
        schema = export_schema(
            model_class,
            normalize=normalize,
            inline_defs=inline_defs,
        )
        output_file = output_path / f"{name}.json"

        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)

        exported_files.append(output_file)
        print(f"Exported: {output_file}")

    return exported_files


def main() -> int:
    """CLI entry point for schema export."""
    parser = argparse.ArgumentParser(
        description="Export flow360-schema Pydantic models to JSON Schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available models
    python -m flow360_schema.export --list

    # Export a specific model to stdout
    python -m flow360_schema.export Simulation

    # Export a specific model to a file
    python -m flow360_schema.export Simulation -o simulation.schema.json

    # Export all models to a directory
    python -m flow360_schema.export --all -o dist/schemas/
        """,
    )

    parser.add_argument("model", nargs="?", help="Name of the model to export")
    parser.add_argument("--list", "-l", action="store_true", help="List all available models")
    parser.add_argument("--all", "-a", action="store_true", help="Export all models")
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--title", "-t", help="Custom title for the schema (single model only)")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip schema normalization (output Pydantic's native format)",
    )
    parser.add_argument(
        "--inline-defs",
        action="store_true",
        help="Inline all definitions (remove definitions block)",
    )

    args = parser.parse_args()

    models = get_exportable_models()

    if args.list:
        if not models:
            print("No exportable models found.")
            print("Models should be defined in flow360_schema.models")
            return 0

        print("Available models:")
        for name in sorted(models.keys()):
            print(f"  - {name}")
        return 0

    if args.all:
        if not args.output:
            print("Error: --all requires --output directory", file=sys.stderr)
            return 1

        normalize = not args.no_normalize
        exported = export_all_schemas(args.output, normalize=normalize, inline_defs=args.inline_defs)
        print(f"\nExported {len(exported)} schema(s) to {args.output}")
        if normalize:
            print("(normalized for common-schema compatibility)")
        return 0

    if not args.model:
        parser.print_help()
        return 0

    if args.model not in models:
        print(f"Error: Model '{args.model}' not found.", file=sys.stderr)
        print(f"Available models: {', '.join(sorted(models.keys()))}", file=sys.stderr)
        return 1

    model_class = models[args.model]
    normalize = not args.no_normalize
    schema = export_schema(model_class, title=args.title, normalize=normalize, inline_defs=args.inline_defs)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"Exported: {output_path}")
    else:
        print(json.dumps(schema, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
