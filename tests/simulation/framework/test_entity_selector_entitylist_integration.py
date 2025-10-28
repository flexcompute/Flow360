import os

import pytest

import flow360 as fl
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2


@pytest.fixture(autouse=True)
def _change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def _load_local_vm():
    return VolumeMeshV2.from_local_storage(
        mesh_id="vm-aa3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=os.path.join(os.path.dirname(__file__), "data", "airplane_volume_mesh"),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id="vm-aa3bb31e-2f85-4504-943c-7788d91c1ab0",
                name="TEST",
                cloud_path_prefix="/",
                status="completed",
            )
        ),
    )


def test_direct_assignment_selector_and_entity_registry_index():
    vm = _load_local_vm()
    # Ensure registry available for __getitem__
    vm.internal_registry = vm._entity_info.get_registry(vm.internal_registry)

    with fl.SI_unit_system:
        all_wings = Surface.match("*Wing", name="all_wings")
        wall = Wall(entities=[all_wings, vm["fluid/leftWing"]])
        # Object-level assertions (before any serialization/validation)
        assert wall.entities.selectors and len(wall.entities.selectors) == 1
        assert wall.entities.selectors[0].target_class == "Surface"
        assert wall.entities.selectors[0].name == "all_wings"
        assert wall.entities.stored_entities and len(wall.entities.stored_entities) == 1
        assert all(
            e.private_attribute_entity_type_name == "Surface" for e in wall.entities.stored_entities
        )

        freestream = fl.Freestream(
            entities=[vm["fluid/farfield"]]
        )  # Legacy entity assignment syntax

        fuselage = Surface.match("flu*fuselage", name="fuselage")

        nothing_surface = Surface.match("nothing", name="nothing")

        # wall_fuselage = Wall(
        #     entities=[fuselage, nothing_surface], use_wall_function=True  # List of EntitySelectors

        # )

        params = fl.SimulationParams(models=[wall, freestream])

    # Fill in project_entity_info to provide selector database
    params = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    # Full validate path (includes resolve_selectors + materialize)
    validated, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level=None,
    )

    assert not errors, f"Unexpected validation errors: {errors}"

    # Ensure at least one entity exists; selector append after explicit
    entities = validated.models[0].entities.stored_entities
    assert len(entities) == 2  # Selector resolved to leftWing and rightWing
    assert entities[0].name == "fluid/leftWing"
    assert entities[1].name == "fluid/rightWing"
    print(">>> entities: ", [entity.name for entity in entities])

    # Legacy
    entities = validated.models[1].entities.stored_entities
    assert len(entities) == 1
    assert entities[0].name == "fluid/farfield"

    # Pure selectors
    # entities = validated.models[2].entities.stored_entities
    # assert len(entities) == 1
    # assert entities[0].name == "fluid/fuselage"
