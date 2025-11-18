import copy
import json
import os

import pytest

import flow360 as fl
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.framework.entity_selector import (
    EntitySelector,
    Predicate,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.services_utils import strip_selector_matches_inplace
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2


@pytest.fixture(autouse=True)
def _change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def _load_local_vm():
    """Fixture to load a local volume mesh for testing."""
    return VolumeMeshV2.from_local_storage(
        mesh_id="vm-aa3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "..", "framework", "data", "airplane_volume_mesh"
        ),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id="vm-aa3bb31e-2f85-4504-943c-7788d91c1ab0",
                name="TEST",
                cloud_path_prefix="/",
                status="completed",
            )
        ),
    )


def _load_json(path_from_tests_dir: str) -> dict:
    """Helper to load a JSON file from the tests/simulation directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "..", path_from_tests_dir), "r", encoding="utf-8") as file:
        return json.load(file)


def test_validate_model_expands_selectors_and_preserves_them():
    """
    Test: End-to-end validation of a mixed entity/selector list.
    - Verifies that `validate_model` expands selectors into `stored_entities`.
    - Verifies that the original `selectors` list is preserved for future edits.
    - Verifies that the process is idempotent.
    """
    vm = _load_local_vm()
    vm.internal_registry = vm._entity_info.get_registry(vm.internal_registry)

    with fl.SI_unit_system:
        all_wings_selector = Surface.match("*Wing", name="all_wings")
        fuselage_selector = Surface.match("flu*fuselage", name="fuselage")
        wall_with_mixed_entities = Wall(entities=[all_wings_selector, vm["fluid/leftWing"]])
        wall_with_only_selectors = Wall(entities=[fuselage_selector])
        freestream = fl.Freestream(entities=[vm["fluid/farfield"]])
        params = fl.SimulationParams(
            models=[wall_with_mixed_entities, wall_with_only_selectors, freestream]
        )

    params_with_cache = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    validated, errors, _ = validate_model(
        params_as_dict=params_with_cache.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    assert not errors, f"Unexpected validation errors: {errors}"

    # Verify expansion: explicit entity + selector results
    expanded_entities1 = validated.models[0].entities.stored_entities
    assert len(expanded_entities1) == 2
    assert expanded_entities1[0].name == "fluid/leftWing"
    assert expanded_entities1[1].name == "fluid/rightWing"

    # Verify pure selector expansion
    expanded_entities2 = validated.models[1].entities.stored_entities
    assert len(expanded_entities2) == 1
    assert expanded_entities2[0].name == "fluid/fuselage"

    # Verify selectors are preserved
    assert validated.models[0].entities.selectors == [all_wings_selector]
    assert validated.models[1].entities.selectors == [fuselage_selector]

    # Verify idempotency
    validated_dict = validated.model_dump(mode="json", exclude_none=True)
    validated_again, errors, _ = validate_model(
        params_as_dict=copy.deepcopy(validated_dict),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    assert not errors, "Validation failed on the second pass"
    assert compare_values(
        validated.model_dump(mode="json"), validated_again.model_dump(mode="json")
    )


def test_validate_model_materializes_dict_and_preserves_selectors():
    """
    Test: `validate_model` correctly materializes entity dicts into objects
    while preserving the original selectors from a raw dictionary input.
    """
    params = _load_json("data/geometry_grouped_by_file/simulation.json")

    # Inject a selector into the params dict and assign all entities to a Wall
    # to satisfy the boundary condition validation.
    selector_dict = {
        "target_class": "Surface",
        "name": "some_selector_name",
        "logic": "AND",
        "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
    }
    outputs = params.get("outputs") or []
    entities = outputs[0].get("entities") or {}
    entities["selectors"] = [selector_dict]
    entities["stored_entities"] = []  # Start with no materialized entities

    # Assign all boundaries to a default wall to pass validation
    all_boundaries_selector = {
        "target_class": "Surface",
        "name": "all_boundaries",
        "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
    }
    params["models"].append(
        {
            "type": "Wall",
            "name": "DefaultWall",
            "entities": {"selectors": [all_boundaries_selector]},
        }
    )

    validated, errors, _ = validate_model(
        params_as_dict=params,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
    )
    assert not errors, f"Unexpected validation errors: {errors}"

    # Verify materialization
    materialized_entities = validated.outputs[0].entities.stored_entities
    assert materialized_entities and all(isinstance(e, Surface) for e in materialized_entities)
    assert len(materialized_entities) > 0

    # Verify selectors are preserved after materialization
    preserved_selectors = validated.outputs[0].entities.selectors
    assert len(preserved_selectors) == 1
    assert preserved_selectors[0].model_dump(exclude_none=True) == selector_dict


def test_validate_model_deduplicates_non_point_entities():
    """
    Test: `validate_model` deduplicates non-Point entities based on (type, id).
    """
    params = {
        "version": "25.7.6b0",
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "o1",
                "output_fields": ["Cp"],
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
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []}
        },
        "unit_system": {"name": "SI"},
    }

    validated, errors, _ = validate_model(
        params_as_dict=params, validated_by=ValidationCalledBy.LOCAL, root_item_type="Case"
    )
    assert not errors
    final_entities = validated.outputs[0].entities.stored_entities
    assert len(final_entities) == 1
    assert final_entities[0].name == "wing"


def test_validate_model_does_not_deduplicate_point_entities():
    """
    Test: `validate_model` preserves duplicate Point entities.
    """
    params = {
        "version": "25.7.6b0",
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
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
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []}
        },
        "unit_system": {"name": "SI"},
    }

    validated, errors, _ = validate_model(
        params_as_dict=params, validated_by=ValidationCalledBy.LOCAL, root_item_type="Case"
    )
    assert not errors
    final_entities = validated.outputs[0].entities.stored_entities
    assert len(final_entities) == 2
    assert all(e.name == "p1" for e in final_entities)


def test_validate_model_shares_entity_instances_across_lists():
    """
    Test: `validate_model` uses a global cache to share entity instances,
    ensuring that an entity with the same ID is the same Python object everywhere.
    """
    entity_dict = {
        "name": "s",
        "private_attribute_entity_type_name": "Surface",
        "private_attribute_id": "s-1",
    }
    params = {
        "version": "25.7.6b0",
        "unit_system": {"name": "SI"},
        "operating_condition": {"type_name": "AerospaceCondition", "velocity_magnitude": 10},
        "models": [
            {"type": "Wall", "name": "Wall", "entities": {"stored_entities": [entity_dict]}}
        ],
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "o3",
                "output_fields": ["Cp"],
                "entities": {"stored_entities": [entity_dict]},
            }
        ],
        "private_attribute_asset_cache": {
            "project_entity_info": {"type_name": "SurfaceMeshEntityInfo", "boundaries": []}
        },
    }

    validated, errors, _ = validate_model(
        params_as_dict=params, validated_by=ValidationCalledBy.LOCAL, root_item_type="Case"
    )
    assert not errors
    entity_in_model = validated.models[0].entities.stored_entities[0]
    entity_in_output = validated.outputs[0].entities.stored_entities[0]
    assert entity_in_model is entity_in_output


def test_strip_selector_matches_preserves_semantics_end_to_end():
    """
    simulation.json -> expand -> mock submit (strip) -> read back -> compare stored_entities
    Ensures stripping selector-matched entities before upload does not change semantics.

    Derivation notes (for future readers):
    - We inject a mixed EntityList (handpicked + selector with overlap) into outputs[0].entities.
    - Baseline: validate_model expands selectors and materializes entities.
    - Mock submit: strip_selector_matches_inplace removes selector matches from stored_entities.
    - Read back: validate_model expands selectors again → final entities must equal baseline.
    """
    # Use a large, real geometry with many faces
    params = _load_json("../data/geo-fcbe1113-a70b-43b9-a4f3-bbeb122d64fb/simulation.json")

    # Set face grouping tag so selector operates on faceId groups
    pei = params["private_attribute_asset_cache"]["project_entity_info"]
    pei["face_group_tag"] = "faceId"
    # Remove obsolete/unknown meshing defaults to avoid validation noise in Case-level
    params.get("meshing", {}).get("defaults", {}).pop("geometry_tolerance", None)

    # Build mixed EntityList with overlap under outputs[0].entities
    outputs = params.get("outputs") or []
    assert outputs, "Test fixture lacks outputs"
    entities = outputs[0].get("entities") or {}
    entities["stored_entities"] = [
        {
            "private_attribute_entity_type_name": "Surface",
            "name": "body00001_face00001",
            "private_attribute_id": "body00001_face00001",
        },
        {
            "private_attribute_entity_type_name": "Surface",
            "name": "body00001_face00014",
            "private_attribute_id": "body00001_face00014",
        },
    ]
    entities["selectors"] = [
        {
            "target_class": "Surface",
            "name": "some_overlap",
            "children": [
                {
                    "attribute": "name",
                    "operator": "any_of",
                    "value": ["body00001_face00001", "body00001_face00002"],
                }
            ],
        }
    ]
    outputs[0]["entities"] = entities
    params["outputs"] = outputs

    # Ensure models contain a DefaultWall that matches all to satisfy BC validation
    all_boundaries_selector = {
        "target_class": "Surface",
        "name": "all_boundaries",
        "children": [{"attribute": "name", "operator": "matches", "value": "*"}],
    }
    params.setdefault("models", []).append(
        {
            "type": "Wall",
            "name": "DefaultWall",
            "entities": {"selectors": [all_boundaries_selector]},
        }
    )

    # Baseline expansion + materialization
    validated, errors, _ = validate_model(
        params_as_dict=params,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
    )
    assert not errors, f"Unexpected validation errors: {errors}"

    baseline_entities = validated.outputs[0].entities.stored_entities  # type: ignore[index]
    baseline_names = sorted(
        [f"{e.private_attribute_entity_type_name}:{e.name}" for e in baseline_entities]
    )

    # Mock submit (strip selector-matched)
    upload_dict = strip_selector_matches_inplace(
        validated.model_dump(mode="json", exclude_none=True)
    )

    # Assert what remains in the upload_dict after stripping selector matches
    upload_entities = (
        upload_dict.get("outputs", [])[0].get("entities", {}).get("stored_entities", [])
    )
    upload_names = sorted(
        [f"{d.get('private_attribute_entity_type_name')}:{d.get('name')}" for d in upload_entities]
    )
    expected_remaining = ["Surface:body00001_face00014"]
    assert upload_names == expected_remaining, (
        "Unexpected remaining stored_entities in upload_dict after stripping\n"
        + f"Remaining: {upload_names}\n"
        + f"Expected : {expected_remaining}\n"
    )

    # Read back and expand again
    validated2, errors2, _ = validate_model(
        params_as_dict=upload_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
    )
    assert not errors2, f"Unexpected validation errors on read back: {errors2}"
    post_entities = validated2.outputs[0].entities.stored_entities  # type: ignore[index]
    post_names = sorted([f"{e.private_attribute_entity_type_name}:{e.name}" for e in post_entities])

    # Show both sides for easy visual inspection
    assert baseline_names == post_names, (
        "Entity list mismatch at outputs[0].entities\n"
        + f"Baseline: {baseline_names}\n"
        + f"Post    : {post_names}\n"
    )

    # Sanity: intended overlap surfaced in baseline
    baseline_only = [s.split(":", 1)[1] for s in baseline_names]
    assert "body00001_face00001" in baseline_only and "body00001_face00002" in baseline_only
    assert "body00001_face00014" in baseline_only
