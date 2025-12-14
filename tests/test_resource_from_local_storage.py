import os
from typing import List, Optional

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360 import log
from flow360.component.case import Case, CaseMeta
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    PorousMedium,
    Rotation,
)
from flow360.component.simulation.primitives import Box
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.surface_mesh_v2 import SurfaceMeshMetaV2, SurfaceMeshV2
from flow360.component.utils import LocalResourceCache
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2

log.set_logging_level("DEBUG")


class Resource(pd.BaseModel):
    s3_path: str = pd.Field(alias="s3Path")
    user_id: str = pd.Field(alias="userId")
    id: str
    name: str
    type: str
    project_id: str = pd.Field(alias="projectId")
    parent_id: Optional[str] = pd.Field(default=None, alias="parentId")
    parent_folder_id: Optional[str] = pd.Field(alias="parentFolderId")


class Path(pd.BaseModel):
    geometry: Optional[Resource] = None
    surface_mesh: Optional[Resource] = pd.Field(None, alias="surfaceMesh")
    volume_mesh: Resource = pd.Field(alias="volumeMesh")
    cases: List[Resource]


class CaseData(pd.BaseModel):
    id: str
    name: str
    path: Path


class ResourceData(pd.BaseModel):
    cases: List[CaseData]


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def pre_upload_change_reflection_vm(volume_mesh: VolumeMeshV2):
    # Test Volume Mesh

    with SI_unit_system:
        volume_mesh["blk-1"].axis = (3, 0, 4)
        volume_mesh["blk-1"].center = (3, 0, 4)
        my_box = Box(
            name="my_box_default",
            center=(1, 2, 3),
            size=(2, 2, 3),
            angle_of_rotation=20 * u.deg,
            axis_of_rotation=(1, 0, 0),
        )
        params = SimulationParams(
            models=[
                Rotation(volumes=[volume_mesh["blk-1"]], spec=AngleExpression("123*t")),
                PorousMedium(
                    volumes=my_box,
                    darcy_coefficient=(1, 1, 1),
                    forchheimer_coefficient=(0.1, 0.1, 0.2),
                ),
            ]
        )
    new_param = set_up_params_for_uploading(
        root_asset=volume_mesh,
        length_unit=1 * u.m,
        params=params,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )
    recorded_zone = new_param.private_attribute_asset_cache.project_entity_info.zones[0]
    assert recorded_zone.axis == (0.6, 0.0, 0.8)
    assert (recorded_zone.center == (3, 0.0, 4) * u.m).all()
    assert my_box in new_param.private_attribute_asset_cache.project_entity_info.draft_entities


def pre_upload_change_reflection_geo(geometry: Geometry):
    # Ensure same behavior as from_cloud()
    geometry.internal_registry = geometry._entity_info.get_persistent_entity_registry(
        geometry.internal_registry
    )
    geometry.group_bodies_by_tag("groupByFile")
    geometry.group_faces_by_tag("faceName")
    # Renaming (must happen before creating the draft copy).
    geometry.rename_surfaces("fuselage", "main_boundary")
    geometry.rename_body_groups("my_file.csm", "main_body")

    with create_draft(new_run_from=geometry) as draft:
        with SI_unit_system:
            # Coordinate systems replace body-group transformations.
            cs = CoordinateSystem(name="main_body_cs", scale=(1.0, 2.0, 3.0))
            main_body = draft.body_groups["main_body"]
            main_body_entity_id = main_body.private_attribute_id
            draft.coordinate_systems.assign(entities=main_body, coordinate_system=cs)

            my_box = Box(
                name="my_box_default",
                center=(1, 2, 3),
                size=(2, 2, 3),
                angle_of_rotation=20 * u.deg,
                axis_of_rotation=(1, 0, 0),
            )
            params = SimulationParams(
                models=[
                    PorousMedium(
                        volumes=my_box,
                        darcy_coefficient=(1, 1, 1),
                        forchheimer_coefficient=(0.1, 0.1, 0.2),
                    ),
                ]
            )

        new_param = set_up_params_for_uploading(
            root_asset=geometry,
            length_unit=1 * u.m,
            params=params,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

    assert new_param.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceName"
    assert (
        new_param.private_attribute_asset_cache.project_entity_info.body_group_tag == "groupByFile"
    )
    assert (
        new_param.private_attribute_asset_cache.project_entity_info.grouped_faces[0][1].name
        == "main_boundary"
    )
    assert (
        new_param.private_attribute_asset_cache.project_entity_info.grouped_bodies[1][0].name
        == "main_body"
    )
    status = new_param.private_attribute_asset_cache.coordinate_system_status
    assert status is not None
    assert status.assignments == [
        CoordinateSystemAssignmentGroup(
            coordinate_system_id=cs.private_attribute_id,
            entities=[
                CoordinateSystemEntityRef(
                    entity_type="GeometryBodyGroup", entity_id=main_body_entity_id
                )
            ],
        )
    ]


def test_resources_from_local_storage_geo():

    resource_data_dict = {
        "cases": [
            {
                "id": "case-11111111-1111-1111-1111-111111111111",
                "name": "Case_alpha=5",
                "path": {
                    "geometry": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "name": "simple-airplane",
                        "type": "Geometry",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "surfaceMesh": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "name": "SurfaceMesh_v1",
                        "type": "SurfaceMesh",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentId": "geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "volumeMesh": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                        "name": "VolumeMesh_v1",
                        "type": "VolumeMesh",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentId": "sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "cases": [
                        {
                            "s3Path": "s3://case-bucket/users/user-29083u29irfjsdkfns/case-11111111-1111-1111-1111-111111111111",
                            "userId": "user-29083u29irfjsdkfns",
                            "id": "case-11111111-1111-1111-1111-111111111111",
                            "name": "Case_alpha=5",
                            "type": "Case",
                            "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                            "parentId": "vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                            "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                        }
                    ],
                },
            },
            {
                "id": "case-2222222222-2222-2222-2222-2222222222",
                "name": "Case_alpha=0",
                "path": {
                    "geometry": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "name": "simple-airplane",
                        "type": "Geometry",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "surfaceMesh": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "name": "SurfaceMesh_v1",
                        "type": "SurfaceMesh",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentId": "geo-e5c01a98-2180-449e-b255-d60162854a83",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "volumeMesh": {
                        "s3Path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                        "userId": "user-29083u29irfjsdkfns",
                        "id": "vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                        "name": "VolumeMesh_v1",
                        "type": "VolumeMesh",
                        "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                        "parentId": "sm-3a7eb4c4-e8c0-4664-b85c-255fe23a3474",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "cases": [
                        {
                            "s3Path": "s3://case-bucket/users/user-29083u29irfjsdkfns/case-2222222222-2222-2222-2222-2222222222",
                            "userId": "user-29083u29irfjsdkfns",
                            "id": "case-2222222222-2222-2222-2222-2222222222",
                            "name": "Case_alpha=0",
                            "type": "Case",
                            "projectId": "prj-ca07aab4-c0f8-4a0c-b07d-67a3804d8dab",
                            "parentId": "vm-93a5dad9-a54c-4db9-a8ab-e22a976bb27a",
                            "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                        }
                    ],
                },
            },
        ]
    }

    resource_data = ResourceData(**resource_data_dict)

    cache = LocalResourceCache()

    for case_meta in resource_data.cases:
        vm_meta = case_meta.path.volume_mesh
        case = Case.from_local_storage(
            local_storage_path=os.path.join("data", case_meta.id),
            meta_data=CaseMeta(
                **local_metadata_builder(
                    id=case_meta.id,
                    name=case_meta.name,
                    parent_id=case_meta.path.volume_mesh.id,
                    case_mesh_id=case_meta.path.volume_mesh.id,
                    cloud_path_prefix=case_meta.path.cases[-1].s3_path.rsplit("/", 1)[0],
                )
            ),
        )
        cache.add(case)

        vm = VolumeMeshV2.from_local_storage(
            mesh_id=vm_meta.id,
            local_storage_path=os.path.join("data", vm_meta.id),
            meta_data=VolumeMeshMetaV2(
                **local_metadata_builder(
                    id=vm_meta.id,
                    name=vm_meta.name,
                    cloud_path_prefix=vm_meta.s3_path.rsplit("/", 1)[0],
                )
            ),
        )
        cache.add(vm)
        pre_upload_change_reflection_vm(volume_mesh=vm)

        sm_meta = case_meta.path.surface_mesh
        if sm_meta is not None:
            sm = SurfaceMeshV2.from_local_storage(
                local_storage_path=os.path.join("data", sm_meta.id),
                meta_data=SurfaceMeshMetaV2(
                    **local_metadata_builder(
                        id=sm_meta.id,
                        name=sm_meta.name,
                        cloud_path_prefix=sm_meta.s3_path.rsplit("/", 1)[0],
                    )
                ),
            )
            cache.add(sm)

        geo_meta = case_meta.path.geometry
        if geo_meta is not None:
            geo = Geometry.from_local_storage(
                geometry_id=geo_meta.id,
                local_storage_path=os.path.join("data", geo_meta.id),
                meta_data=GeometryMeta(
                    **local_metadata_builder(
                        id=geo_meta.id,
                        name=geo_meta.name,
                        cloud_path_prefix=geo_meta.s3_path.rsplit("/", 1)[0],
                        status="processed",
                    )
                ),
            )
            pre_upload_change_reflection_geo(geometry=geo)
            cache.add(geo)

    cases = [cache[case.id] for case in resource_data.cases]

    assert cases[0].volume_mesh.info.status.value == "completed"
    for boundary in ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]:
        assert boundary in cases[0].volume_mesh.boundary_names

    for case in cases:
        assert case.params.time_stepping.type_name == "Steady"


def test_resource_from_local_storage_vm():
    resource_data_dict = {
        "cases": [
            {
                "id": "case-bbf9a4dc-f5f7-42ee-bfe8-8905d9e45386",
                "name": "Case_DrivAer 225M",
                "path": {
                    "volumeMesh": {
                        "s3Path": "s3://flow360meshes/users/XXXXXXX/vm-1cfdec99-3ce3-428c-85f8-2054812b2ddc",
                        "userId": "XXXXXXX",
                        "id": "vm-1cfdec99-3ce3-428c-85f8-2054812b2ddc",
                        "name": "DrivAer",
                        "type": "VolumeMesh",
                        "projectId": "prj-ef977140-66c7-4b10-ab51-c7ab821afc4b",
                        "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                    },
                    "cases": [
                        {
                            "s3Path": "s3://flow360cases/users/XXXXXXX/case-bbf9a4dc-f5f7-42ee-bfe8-8905d9e45386",
                            "userId": "XXXXXXX",
                            "id": "case-bbf9a4dc-f5f7-42ee-bfe8-8905d9e45386",
                            "name": "Case_DrivAer 225M",
                            "type": "Case",
                            "projectId": "prj-ef977140-66c7-4b10-ab51-c7ab821afc4b",
                            "parentId": "vm-1cfdec99-3ce3-428c-85f8-2054812b2ddc",
                            "parentFolderId": "folder-86a7bcb3-fb9d-44ca-acd5-c6744b15f582",
                        }
                    ],
                },
            }
        ]
    }

    resource_data = ResourceData(**resource_data_dict)

    cache = LocalResourceCache()

    for case_meta in resource_data.cases:
        vm_meta = case_meta.path.volume_mesh
        case = Case.from_local_storage(
            local_storage_path=os.path.join("data", case_meta.id),
            meta_data=CaseMeta(
                **local_metadata_builder(
                    id=case_meta.id,
                    name=case_meta.name,
                    parent_id=case_meta.path.volume_mesh.id,
                    case_mesh_id=case_meta.path.volume_mesh.id,
                    cloud_path_prefix=case_meta.path.cases[-1].s3_path.rsplit("/", 1)[0],
                )
            ),
        )
        cache.add(case)

        vm = VolumeMeshV2.from_local_storage(
            mesh_id=vm_meta.id,
            local_storage_path=os.path.join("data", vm_meta.id),
            meta_data=VolumeMeshMetaV2(
                **local_metadata_builder(
                    id=vm_meta.id,
                    name=vm_meta.name,
                    cloud_path_prefix=vm_meta.s3_path.rsplit("/", 1)[0],
                )
            ),
        )
        cache.add(vm)

        sm_meta = case_meta.path.surface_mesh
        if sm_meta is not None:
            sm = SurfaceMeshV2.from_local_storage(
                local_storage_path=os.path.join("data", sm_meta.id),
                meta_data=SurfaceMeshMetaV2(
                    **local_metadata_builder(
                        id=sm_meta.id,
                        name=sm_meta.name,
                        cloud_path_prefix=sm_meta.s3_path.rsplit("/", 1)[0],
                    )
                ),
            )
            cache.add(sm)

        geo_meta = case_meta.path.geometry
        if geo_meta is not None:
            geo = Geometry.from_local_storage(
                geometry_id=geo_meta.id,
                local_storage_path=os.path.join("data", geo_meta.id),
                meta_data=GeometryMeta(
                    **local_metadata_builder(
                        id=geo_meta.id,
                        name=geo_meta.name,
                        cloud_path_prefix=geo_meta.s3_path.rsplit("/", 1)[0],
                        status="processed",
                    )
                ),
            )
            cache.add(geo)

    cases = [cache[case.id] for case in resource_data.cases]

    assert cases[0].volume_mesh.info.status.value == "completed"
    for boundary in ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]:
        assert boundary in cases[0].volume_mesh.boundary_names
