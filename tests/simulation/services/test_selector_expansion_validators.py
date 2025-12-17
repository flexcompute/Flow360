import os

import pytest

import flow360 as fl
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.framework.entity_selector import Predicate
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    ForcePerArea,
    Rotation,
)
from flow360.component.simulation.primitives import Cylinder, GenericVolume, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2


@pytest.fixture(autouse=True)
def _change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def _load_airplane_vm():
    """Load a local volume mesh asset used by selector expansion tests."""
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


def _load_mock_volume_mesh_with_single_zone():
    """Load a local volume mesh asset that contains a GenericVolume zone named blk-1."""
    return VolumeMeshV2.from_local_storage(
        mesh_id="vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
        local_storage_path=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
        ),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id="vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                name="TEST",
                cloud_path_prefix="/",
                status="completed",
            )
        ),
    )


def _extract_error_messages(errors):
    return [err.get("msg", "") for err in (errors or [])]


def test_duplicate_entities_in_models_detects_selector_overlap():
    vm = _load_airplane_vm()
    vm.internal_registry = vm._entity_info.get_persistent_entity_registry(vm.internal_registry)

    with fl.SI_unit_system:
        all_surfaces = Surface.match("*", name="all_surfaces")
        wings = Surface.match("*Wing", name="wings")
        params = fl.SimulationParams(
            models=[Wall(name="wallAll", entities=[all_surfaces]), Wall(entities=[wings])]
        )

    params_with_cache = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    _validated, errors, _warnings = validate_model(
        params_as_dict=params_with_cache.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    messages = "\n".join(_extract_error_messages(errors))
    assert "appears multiple times" in messages
    assert "Surface entity" in messages


def test_duplicate_surface_usage_detects_selector_overlap():
    vm = _load_airplane_vm()
    vm.internal_registry = vm._entity_info.get_persistent_entity_registry(vm.internal_registry)

    with fl.SI_unit_system:
        wall_all = Wall(entities=[Surface.match("*", name="all_boundaries")])
        output_1 = fl.SurfaceOutput(
            output_fields=["Cp"], entities=[Surface.match("*Wing", name="wings")]
        )
        output_2 = fl.SurfaceOutput(
            output_fields=["Cp"], entities=[Surface.match("fluid/leftWing", name="leftWing")]
        )
        params = fl.SimulationParams(models=[wall_all], outputs=[output_1, output_2])

    params_with_cache = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    _validated, errors, _warnings = validate_model(
        params_as_dict=params_with_cache.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    messages = "\n".join(_extract_error_messages(errors))
    assert "used in multiple" in messages
    assert "SurfaceOutput" in messages


def test_parent_volume_is_rotating_allows_selector_based_volume_sets():
    """
    Regression test: Rotation parent_volume validation must see selector-expanded entities.

    Previously, if a Rotation model specified volumes only via selectors (stored_entities empty),
    check_parent_volume_is_rotating would incorrectly ignore those entities.
    """
    vm = _load_mock_volume_mesh_with_single_zone()

    # Ensure the persistent zone has required rotation attributes. set_up_params_for_uploading will
    # reflect these edits into the asset cache used during validate_model.
    with fl.SI_unit_system:
        vm["blk-1"].axis = (0, 0, 1)
        vm["blk-1"].center = (0, 0, 0) * fl.u.m

        outer_rotation = Rotation(
            volumes=[GenericVolume.match("blk-1", name="outer_zone_selector")],
            spec=fl.AngleExpression("sin(t)"),
            name="outerRotation",
        )
        inner_cylinder = Cylinder(
            name="innerCylinder",
            axis=(0, 0, 1),
            center=(0, 0, 0),
            height=1.0,
            outer_radius=1.0,
        )
        inner_rotation = Rotation(
            volumes=[inner_cylinder],
            spec=fl.AngleExpression("-2*sin(t)"),
            parent_volume=vm["blk-1"],
            name="innerRotation",
        )
        wall_all = Wall(entities=vm["*"])
        params = fl.SimulationParams(models=[wall_all, outer_rotation, inner_rotation])

    params_with_cache = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    _validated, errors, _warnings = validate_model(
        params_as_dict=params_with_cache.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    assert not errors, f"Unexpected validation errors: {errors}"


def test_duplicate_actuator_disk_cylinder_names_detects_selector_overlap():
    vm = _load_mock_volume_mesh_with_single_zone()

    with fl.SI_unit_system:
        fpa = ForcePerArea(radius=[0, 1, 2], thrust=[1, 1, 1], circumferential=[0, 0, 0])
        cylinder = Cylinder(
            name="dupCylinder",
            axis=(1, 0, 0),
            center=(0, 0, 0),
            height=1.0,
            outer_radius=1.0,
        )
        # Cylinders are draft entities and EntitySelector currently does not support Cylinder
        # as a target_class. This regression test ensures the validator still catches duplicates
        # across multiple ActuatorDisk instances.
        ad1 = ActuatorDisk(volumes=[cylinder], force_per_area=fpa)
        ad2 = ActuatorDisk(volumes=[cylinder], force_per_area=fpa)
        wall_all = Wall(entities=vm["*"])
        # Build a valid SimulationParams first, then inject the duplicate model to exercise
        # validate_model() error reporting on dictionary inputs.
        params = fl.SimulationParams(models=[wall_all, ad1])
        params.models.append(ad2)

    params_with_cache = set_up_params_for_uploading(
        vm, 1 * fl.u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    _validated, errors, _warnings = validate_model(
        params_as_dict=params_with_cache.model_dump(mode="json", exclude_none=True),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    messages = "\n".join(_extract_error_messages(errors))
    assert "ActuatorDisk cylinder name `dupCylinder`" in messages
