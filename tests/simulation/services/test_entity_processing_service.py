import copy
import json
import os

import pytest

import flow360 as fl
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.entity_selector import (
    EntitySelector,
    Predicate,
    SurfaceSelector,
)
from flow360.component.simulation.framework.param_utils import AssetCache
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


def test_validate_model_keeps_selectors_unexpanded():
    """
    Test: End-to-end validation with delayed selector expansion.
    - Verifies that `validate_model` does NOT expand selectors into `stored_entities`.
    - Verifies that explicitly specified entities remain in `stored_entities`.
    - Verifies that `selectors` are preserved for future expansion (e.g., translation).
    - Verifies that the process is idempotent.
    """
    vm = _load_local_vm()
    vm.internal_registry = vm._entity_info.get_persistent_entity_registry(vm.internal_registry)

    with fl.SI_unit_system:
        all_wings_selector = SurfaceSelector(name="all_wings").match("*Wing")
        fuselage_selector = SurfaceSelector(name="fuselage").match("flu*fuselage")
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

    # Verify delayed expansion: selectors are NOT expanded into stored_entities.
    # Note: set_up_params_for_uploading() now strips entities that overlap with selectors so the UI can
    # distinguish selector-implied selections from hand-picked ones. Since `fluid/leftWing` also matches
    # the `*Wing` selector, it is expected to be stripped from stored_entities before validation.
    stored_entities1 = validated.models[0].entities.stored_entities
    assert (
        len(stored_entities1) == 0
    ), "Selector-overlap entities should be stripped prior to upload"

    # Verify pure selector case: stored_entities should be empty
    stored_entities2 = validated.models[1].entities.stored_entities
    assert (
        len(stored_entities2) == 0
    ), "stored_entities should be empty when only selectors are used"

    # Verify selectors are preserved for future expansion (e.g., translation)
    # Selectors are deserialized as EntitySelector (base class), check attributes instead
    assert len(validated.models[0].entities.selectors) == 1
    assert validated.models[0].entities.selectors[0].name == "all_wings"
    assert validated.models[0].entities.selectors[0].target_class == "Surface"

    assert len(validated.models[1].entities.selectors) == 1
    assert validated.models[1].entities.selectors[0].name == "fuselage"
    assert validated.models[1].entities.selectors[0].target_class == "Surface"

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
    Test: `validate_model` correctly materializes explicit entity dicts into objects
    while preserving the original selectors from a raw dictionary input.

    With delayed expansion, selectors are NOT expanded into stored_entities during
    validation. Expansion happens later during translation.
    """
    params = _load_json("data/geometry_grouped_by_file/simulation.json")

    # Inject a selector into the params dict and assign all entities to a Wall
    # to satisfy the boundary condition validation.
    selector_dict = {
        "target_class": "Surface",
        "name": "some_selector_name",
        "logic": "AND",
        "selector_id": "some_selector_id",
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
        "children": [
            {"attribute": "name", "operator": "matches", "value": "*"},
            {"attribute": "name", "operator": "not_matches", "value": "farfield"},
            {"attribute": "name", "operator": "not_matches", "value": "symmetric"},
        ],
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

    # With delayed expansion, stored_entities should remain empty since only selectors were specified
    stored_entities = validated.outputs[0].entities.stored_entities
    assert len(stored_entities) == 0, "stored_entities should be empty with delayed expansion"

    # Verify selectors are preserved
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


def test_strip_selector_matches_removes_selector_overlap():
    """Ensure selector-overlap entities are dropped prior to upload."""
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            outputs=[
                fl.SurfaceOutput(
                    name="surface_output",
                    output_fields=[fl.UserVariable(name="var", value=1)],
                    entities=[
                        Surface(name="front", private_attribute_id="s-1"),
                        Surface(name="rear", private_attribute_id="s-2"),
                        SurfaceSelector(name="front_selector").any_of(["front"]),
                    ],
                )
            ],
            private_attribute_asset_cache=AssetCache(
                project_entity_info=SurfaceMeshEntityInfo(
                    boundaries=[
                        Surface(name="front", private_attribute_id="s-1"),
                        Surface(name="rear", private_attribute_id="s-2"),
                    ]
                )
            ),
        )
    strip_selector_matches_inplace(params)
    assert [entity.name for entity in params.outputs[0].entities.stored_entities] == ["rear"]


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


def test_delayed_expansion_round_trip_preserves_semantics():
    """
    simulation.json -> validate -> round-trip -> compare
    Ensures delayed expansion maintains consistency across round-trips.

    With delayed expansion, selectors are NOT expanded into stored_entities during
    validation. This test verifies:
    - Explicit entities remain in stored_entities
    - Selectors are preserved
    - Round-trip maintains consistency
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
        "children": [
            {"attribute": "name", "operator": "matches", "value": "*"},
            {"attribute": "name", "operator": "not_matches", "value": "farfield"},
        ],
    }
    params.setdefault("models", []).append(
        {
            "type": "Wall",
            "name": "DefaultWall",
            "entities": {"selectors": [all_boundaries_selector]},
        }
    )

    # Baseline validation (with delayed expansion)
    validated, errors, _ = validate_model(
        params_as_dict=params,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
    )
    assert not errors, f"Unexpected validation errors: {errors}"

    # With delayed expansion, stored_entities should only contain explicit entities
    baseline_entities = validated.outputs[0].entities.stored_entities  # type: ignore[index]
    baseline_names = sorted(
        [f"{e.private_attribute_entity_type_name}:{e.name}" for e in baseline_entities]
    )
    # Only explicitly specified entities should be present (NOT selector matches)
    expected_explicit = ["Surface:body00001_face00001", "Surface:body00001_face00014"]
    assert baseline_names == expected_explicit, (
        f"Expected only explicit entities in stored_entities\n"
        f"Got: {baseline_names}\n"
        f"Expected: {expected_explicit}"
    )

    # Verify selectors are preserved
    baseline_selectors = validated.outputs[0].entities.selectors
    assert len(baseline_selectors) == 1
    assert baseline_selectors[0].name == "some_overlap"

    # Round-trip: serialize and re-validate
    round_trip_dict = validated.model_dump(mode="json", exclude_none=True)
    validated2, errors2, _ = validate_model(
        params_as_dict=round_trip_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Case",
    )
    assert not errors2, f"Unexpected validation errors on round-trip: {errors2}"

    # Verify round-trip consistency
    post_entities = validated2.outputs[0].entities.stored_entities  # type: ignore[index]
    post_names = sorted([f"{e.private_attribute_entity_type_name}:{e.name}" for e in post_entities])
    assert baseline_names == post_names, (
        "Entity list mismatch after round-trip\n"
        + f"Baseline: {baseline_names}\n"
        + f"Post    : {post_names}\n"
    )

    # Verify selectors are still preserved after round-trip
    post_selectors = validated2.outputs[0].entities.selectors
    assert len(post_selectors) == 1
    assert post_selectors[0].name == "some_overlap"
