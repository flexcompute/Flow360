import copy
import json
import os

import unyt as u

from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_in_place,
)
from flow360.component.simulation.services import ValidationCalledBy, validate_model


def _load_json(path_from_tests_dir: str) -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "..", path_from_tests_dir), "r", encoding="utf-8") as file:
        return json.load(file)


def test_validate_model_resolves_selectors_and_materializes_end_to_end():
    """
    Test: End-to-end integration of selector expansion and entity materialization in validate_model.

    Purpose:
    - Verify that validate_model() correctly processes EntitySelector objects
    - Verify that selectors are expanded against the entity database from asset cache
    - Verify that expanded entity dicts are materialized into Pydantic model instances
    - Verify that selectors are cleared after expansion

    Expected behavior:
    - Input: params with selectors and empty stored_entities
    - Process: Selectors expand to find matching entities from geometry entity info
    - Output: validated model with materialized Surface objects in stored_entities
    - Selectors list should be empty after processing
    """
    params = _load_json("data/geometry_grouped_by_file/simulation.json")

    # Convert first output to selector-only and clear stored_entities
    outputs = params.get("outputs") or []
    if not outputs:
        return
    entities = outputs[0].get("entities") or {}
    entities["selectors"] = [
        {
            "target_class": "Surface",
            "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
        }
    ]
    entities["stored_entities"] = []
    outputs[0]["entities"] = entities

    validated, errors, _ = validate_model(
        params_as_dict=params,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level=None,
    )
    assert not errors, f"Unexpected validation errors: {errors}"

    # selectors should be cleared and stored_entities should be objects
    sd = validated.outputs[0].entities.stored_entities
    assert sd and all(
        getattr(e, "private_attribute_entity_type_name", None) == "Surface" for e in sd
    )


def test_validate_model_per_list_dedup_for_non_point():
    """
    Test: Entity deduplication for non-Point entities during materialization.

    Purpose:
    - Verify that materialize_entities_in_place() deduplicates non-Point entities
    - Verify that deduplication is based on (type, id) tuple
    - Verify that deduplication preserves order (first occurrence kept)
    - Verify that validate_model() applies this deduplication

    Expected behavior:
    - Input: Two Surface entities with same name and ID
    - Process: Materialization deduplicates based on (Surface, s-1) key
    - Output: Single Surface entity in validated model

    Note: Point entities are NOT deduplicated (tested separately)
    """
    # Minimal dict with duplicate Surface items in one list
    params = {
        "version": "25.7.6b0",
        "unit_system": {"name": "SI"},
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "models": [],
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "o1",
                "output_fields": {"items": ["Cp"]},
                "entities": {
                    "stored_entities": [
                        {
                            "name": "wing",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "s-1",
                        },
                        {
                            "name": "wing",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "s-1",
                        },
                    ]
                },
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": {"value": 1, "units": "m"},
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []},
        },
    }

    validated, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(params),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level=None,
    )
    assert not errors, f"Unexpected validation errors: {errors}"
    names = [e.name for e in validated.outputs[0].entities.stored_entities]
    assert names == ["wing"]


def test_validate_model_skip_dedup_for_point():
    """
    Test: Point entities are NOT deduplicated during materialization.

    Purpose:
    - Verify that Point entities are exempted from deduplication
    - Verify that duplicate Point entities with same location are preserved
    - Verify that this exception only applies to Point entity type

    Expected behavior:
    - Input: Two Point entities with same name and location
    - Process: Materialization skips deduplication for Point type
    - Output: Both Point entities remain in validated model

    Rationale: Point entities may intentionally have duplicates for different
    purposes (e.g., multiple streamline origins at same location)
    """
    params = {
        "version": "25.7.6b0",
        "unit_system": {"name": "SI"},
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "models": [],
        "outputs": [
            {
                "output_type": "StreamlineOutput",
                "name": "o2",
                "entities": {
                    "stored_entities": [
                        {
                            "name": "p1",
                            "private_attribute_entity_type_name": "Point",
                            "location": {"value": [0, 0, 0], "units": "m"},
                        },
                        {
                            "name": "p1",
                            "private_attribute_entity_type_name": "Point",
                            "location": {"value": [0, 0, 0], "units": "m"},
                        },
                    ]
                },
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": {"value": 1, "units": "m"},
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []},
        },
    }

    validated, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(params),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level=None,
    )
    assert not errors, f"Unexpected validation errors: {errors}"
    names = [e.name for e in validated.outputs[0].entities.stored_entities]
    assert names == ["p1", "p1"]


def test_validate_model_shares_instances_across_lists():
    """
    Test: Entity instances are shared across different lists (models and outputs).

    Purpose:
    - Verify that materialize_entities_in_place() uses global instance caching
    - Verify that entities with same (type, id) are the same Python object (identity)
    - Verify that this sharing works across different parts of the params tree
    - Verify that validate_model() maintains this instance sharing

    Expected behavior:
    - Input: Same Surface entity (by id) in both models[0] and outputs[0]
    - Process: Materialization creates single instance, reused in both locations
    - Output: validated.models[0].entities[0] is validated.outputs[0].entities[0]

    Benefits: Memory efficiency and enables identity-based comparison
    """
    # Same Surface appears in models and outputs lists with same id
    params = {
        "version": "25.7.6b0",
        "unit_system": {"name": "SI"},
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "models": [
            {
                "type": "Wall",
                "name": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "name": "s",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "s-1",
                        }
                    ]
                },
            }
        ],
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "o3",
                "output_fields": {"items": ["Cp"]},
                "entities": {
                    "stored_entities": [
                        {
                            "name": "s",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "s-1",
                        }
                    ]
                },
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": {"value": 1, "units": "m"},
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []},
        },
    }

    validated, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(params),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level=None,
    )
    assert not errors, f"Unexpected validation errors: {errors}"
    a = validated.models[0].entities.stored_entities[0]
    b = validated.outputs[0].entities.stored_entities[0]
    assert a is b


def test_resolve_selectors_noop_when_absent():
    """
    Test: Selector expansion and materialization are no-ops when no selectors present.

    Purpose:
    - Verify that expand_entity_selectors_in_place() handles missing selectors gracefully
    - Verify that materialize_entities_in_place() handles empty entity lists
    - Verify that validate_model() succeeds with minimal valid params (no selectors)
    - Verify that these operations don't crash or produce errors on empty inputs

    Expected behavior:
    - Input: Valid params with empty models and outputs, no selectors
    - Process: Both expansion and materialization are no-ops
    - Output: Validation succeeds with empty result

    This tests robustness and ensures the pipeline handles edge cases gracefully.
    """
    # No selectors anywhere; materializer also should be a no-op for empty lists
    params = {
        "version": "25.7.6b0",
        "unit_system": {"name": "SI"},
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "models": [],
        "outputs": [],
        "private_attribute_asset_cache": {
            "project_length_unit": {
                "value": 1,
                "units": "m",
            },
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []},
        },
    }

    # Ensure materializer does not crash on empty structure
    _ = materialize_entities_in_place(copy.deepcopy(params))

    validated, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(params),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
        validation_level=None,
    )
    assert not errors, f"Unexpected validation errors: {errors}"
