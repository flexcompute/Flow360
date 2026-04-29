import copy
import os

import pytest

import flow360 as fl
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.draft_context.mirror import MirrorPlane, MirrorStatus
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.entity_expansion_utils import (
    expand_entity_list_in_context,
)
from flow360.component.simulation.framework.entity_selector import SurfaceSelector
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.primitives import MirroredSurface, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
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


def test_expand_entity_list_in_context_includes_mirrored_entities_from_mirror_status():
    """Ensure selector expansion can see mirrored entities registered from mirror_status."""
    mirror_plane = MirrorPlane(
        name="plane",
        normal=(1, 0, 0),
        center=(0, 0, 0) * fl.u.m,
        private_attribute_id="mp-1",
    )

    mirrored_surface = MirroredSurface(
        name="front_<mirror>",
        surface_id="s-1",
        mirror_plane_id="mp-1",
        private_attribute_id="ms-1",
    )

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            outputs=[
                fl.SurfaceOutput(
                    name="surface_output",
                    output_fields=[fl.UserVariable(name="var", value=1)],
                    entities=[SurfaceSelector(name="all").match("*")],
                )
            ],
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=True,
                use_geometry_AI=True,
                project_entity_info=SurfaceMeshEntityInfo(
                    boundaries=[Surface(name="front", private_attribute_id="s-1")]
                ),
                mirror_status=MirrorStatus(
                    mirror_planes=[mirror_plane],
                    mirrored_geometry_body_groups=[],
                    mirrored_surfaces=[mirrored_surface],
                ),
            ),
        )

    # Validate schema-level correctness (skip contextual validation since this test doesn't
    # provide full Case-level required fields like meshing/models/operating_condition).
    validated, errors, _warnings = validate_model(
        params_as_dict=params.model_dump(exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
        validation_level=None,
    )
    assert errors is None
    assert validated is not None

    expanded = expand_entity_list_in_context(params.outputs[0].entities, params, return_names=False)
    expanded_type_names = {entity.private_attribute_entity_type_name for entity in expanded}
    assert "Surface" in expanded_type_names
    assert "MirroredSurface" in expanded_type_names
