import os

import pytest

import flow360 as fl
from flow360 import log
from flow360.component.case import CaseMeta
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import Camera, Chart3D
from flow360.plugins.report.uvf_shutter import (
    ActionPayload,
    SetCameraPayload,
    SetFieldPayload,
    SetObjectVisibilityPayload,
    ShutterBatchService,
    TakeScreenshotPayload,
)

log.set_logging_level("DEBUG")


@pytest.fixture
def here():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def cases(here):
    case1 = fl.Case.from_local_storage(
        os.path.join(here, "case1"),
        CaseMeta(
            caseId="case-fce59889-461f-47a1-85d7-b565d0102728",
            name="case1",
            status="completed",
            userId="user-id",
            caseMeshId="vm-8dfedf66-2255-448c-8334-d3232fa739da",
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        ),
    )

    case2 = fl.Case.from_local_storage(
        os.path.join(here, "case2"),
        CaseMeta(
            caseId="case-5cca5633-47a9-44ad-b832-8ede0fad32fe",
            name="case2",
            status="completed",
            userId="user-id",
            caseMeshId="vm-8dfedf66-2255-448c-8334-d3232fa739da",
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        ),
    )

    case3 = fl.Case.from_local_storage(
        os.path.join(here, "case3"),
        CaseMeta(
            caseId="case-a3c58135-1fe5-4ea8-964c-548138782f42",
            name="case3",
            status="completed",
            userId="user-id",
            caseMeshId="vm-8dfedf66-2255-448c-8334-d3232fa739da",
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        ),
    )

    return [case1, case2, case3]


@pytest.fixture
def action_payloads():
    screenshot = ActionPayload(
        action="take-screenshot", payload=TakeScreenshotPayload(file_name="image1.png")
    )
    set_field = ActionPayload(
        action="set-field",
        payload=SetFieldPayload(
            object_id="boundaries",
            field_name="Cp",
            min_max=(-1, 2),
            is_log_scale=False,
        ),
    )

    set_visibility_1 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object1"], visibility=True),
    )

    set_visibility_2 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object2"], visibility=True),
    )

    set_visibility_3 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object3"], visibility=False),
    )

    set_visibility_4 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object4"], visibility=False),
    )

    set_visibility_5 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object5"], visibility=False),
    )

    set_visibility_6 = ActionPayload(
        action="set-object-visibility",
        payload=SetObjectVisibilityPayload(object_ids=["object6"], visibility=False),
    )

    set_camera_1 = ActionPayload(action="set-camera", payload=SetCameraPayload(position=(0, 0, 1)))

    set_camera_2 = ActionPayload(action="set-camera", payload=SetCameraPayload(position=(1, 0, 0)))

    return {
        "screenshot": screenshot,
        "set_field": set_field,
        "set_visibility_1": set_visibility_1,
        "set_visibility_2": set_visibility_2,
        "set_visibility_3": set_visibility_3,
        "set_visibility_4": set_visibility_4,
        "set_visibility_5": set_visibility_5,
        "set_visibility_6": set_visibility_6,
        "set_camera_1": set_camera_1,
        "set_camera_2": set_camera_2,
    }


def test_shutter_requests(cases):
    top_camera = Camera(
        position=(0, 0, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 1, 0),
        dimension=5,
        dimension_dir="width",
    )
    top_camera_slice = Camera(
        position=(0, 0, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 1, 0),
        dimension=10,
        dimension_dir="width",
    )
    side_camera = Camera(
        position=(0, -1, 0),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=5,
        dimension_dir="width",
    )
    side_camera_slice = Camera(
        position=(0, -1, 0),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 1.5),
        up=(0, 0, 1),
        dimension=10,
        dimension_dir="width",
    )
    back_camera = Camera(position=(1, 0, 0), up=(0, 0, 1), dimension=2.5, dimension_dir="width")
    front_camera = Camera(position=(-1, 0, 0), up=(0, 0, 1), dimension=2.5, dimension_dir="width")
    bottom_camera = Camera(
        position=(0, 0, -1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, -1, 0),
        dimension=5,
        dimension_dir="width",
    )
    front_left_bottom_camera = Camera(
        position=(-1, -1, -1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=5,
        dimension_dir="width",
    )
    rear_right_bottom_camera = Camera(
        position=(1, 1, -1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=5,
        dimension_dir="width",
    )
    front_left_top_camera = Camera(
        position=(-1, -1, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=6,
        dimension_dir="width",
    )
    rear_left_top_camera = Camera(
        position=(1, -1, 1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=6,
        dimension_dir="width",
    )
    front_side_bottom_camera = Camera(
        position=(-1, -1, -1),
        look_at=(0, 0, 0),
        pan_target=(1.5, 0, 0),
        up=(0, 0, 1),
        dimension=6,
        dimension_dir="width",
    )

    cameras_geo = [
        top_camera,
        side_camera,
        back_camera,
        bottom_camera,
        front_left_bottom_camera,
        rear_right_bottom_camera,
    ]

    limits_cp = [(-1, 1), (-1, 1), (-1, 1), (-0.3, 0), (-0.3, 0), (-1, 1), (-1, 1), (-1, 1)]
    cameras_cp = [
        front_camera,
        front_left_top_camera,
        side_camera,
        rear_left_top_camera,
        back_camera,
        bottom_camera,
        front_left_bottom_camera,
        rear_right_bottom_camera,
    ]

    exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]

    report = ReportTemplate(
        title="Aerodynamic analysis",
        items=[
            *[
                Chart3D(
                    section_title="Geometry",
                    items_in_row=2,
                    force_new_page=True,
                    show="boundaries",
                    camera=camera,
                    exclude=exclude,
                    fig_name=f"geo_{i}",
                )
                for i, camera in enumerate(cameras_geo)
            ],
            Chart3D(
                section_title="Slice velocity",
                items_in_row=2,
                force_new_page=True,
                show="slices",
                include=["y-slice through moment center"],
                field="velocity",
                limits=(0, 0.18),
                camera=side_camera_slice,
                fig_name="slice_y",
            ),
            Chart3D(
                section_title="Slice velocity",
                items_in_row=2,
                force_new_page=True,
                show="slices",
                include=["z-slice through moment center"],
                field="velocity",
                limits=(0, 0.18),
                camera=top_camera_slice,
                fig_name="slice_z",
            ),
            *[
                Chart3D(
                    section_title="y+",
                    items_in_row=2,
                    show="boundaries",
                    field="yPlus",
                    exclude=exclude,
                    limits=(0, 100),
                    camera=camera,
                    fig_name=f"yplus_{i}",
                    caption=f"limits={(0, 100)}",
                )
                for i, camera in enumerate([top_camera, bottom_camera])
            ],
            *[
                Chart3D(
                    section_title="Cp",
                    items_in_row=2,
                    show="boundaries",
                    field="Cp",
                    exclude=exclude,
                    limits=limits,
                    camera=camera,
                    fig_name=f"cp_{i}",
                    caption=f"limits={limits}",
                )
                for i, (limits, camera) in enumerate(zip(limits_cp, cameras_cp))
            ],
            Chart3D(
                section_title="Q-criterion",
                items_in_row=2,
                force_new_page=True,
                show="qcriterion",
                exclude=exclude,
                field="Mach",
                limits=(0, 0.18),
                fig_name="qcriterion",
            ),
        ],
    )

    requests = []
    service = ShutterBatchService()

    for chart in report.items:
        if isinstance(chart, Chart3D):
            for case in cases:
                req = chart._get_shutter_request(case)
                requests.append(req)
                if not chart._fig_exist(case.id):
                    service.add_request(req)

    for resource, scene in service.requests.items():
        print(resource, len(scene[0].script))

    assert len(requests) == len(report.items) * len(cases)

    for i in range(0, len(requests), len(cases)):
        assert requests[i].resource.id == "case-fce59889-461f-47a1-85d7-b565d0102728"
        assert requests[i + 1].resource.id == "case-5cca5633-47a9-44ad-b832-8ede0fad32fe"
        assert requests[i + 2].resource.id == "case-a3c58135-1fe5-4ea8-964c-548138782f42"

    batch_requests = service.get_batch_requests()
    assert len(batch_requests) == len(cases)
    for i in range(0, len(batch_requests), len(cases)):
        assert batch_requests[i].resource.id == "case-fce59889-461f-47a1-85d7-b565d0102728"
        assert batch_requests[i + 1].resource.id == "case-5cca5633-47a9-44ad-b832-8ede0fad32fe"
        assert batch_requests[i + 2].resource.id == "case-a3c58135-1fe5-4ea8-964c-548138782f42"


def test_merge_visibility_actions_no_merge(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions)


def test_merge_visibility_actions_merge(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["set_field"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions) - 1
    assert set(processed_actions[4].payload.object_ids) == {"object1", "object2"}


def test_merge_visibility_actions_merge_many(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["set_field"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_2"],
        action_payloads["set_visibility_3"],
        action_payloads["set_visibility_4"],
        action_payloads["set_visibility_5"],
        action_payloads["set_visibility_6"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    print(processed_actions)
    assert len(processed_actions) == len(actions) - 4
    assert set(processed_actions[4].payload.object_ids) == {"object1", "object2"}
    assert set(processed_actions[5].payload.object_ids) == {
        "object3",
        "object4",
        "object5",
        "object6",
    }


def test_merge_and_remove_redundant_visibility_actions(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_2"],
        action_payloads["set_visibility_3"],
        action_payloads["set_visibility_4"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_2"],
        action_payloads["set_visibility_3"],
        action_payloads["set_visibility_4"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions) - 4
    assert set(processed_actions[4].payload.object_ids) == {"object1", "object2"}

    optimized_actions = service._remove_redundant_visibility_actions(processed_actions)
    assert len(optimized_actions) == len(actions) - 6


def test_partial_merge_visibility_actions(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_2"],
        action_payloads["set_visibility_3"],
        action_payloads["set_visibility_4"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["set_visibility_3"],
        action_payloads["set_visibility_2"],
        action_payloads["set_visibility_4"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions) - 2
    assert set(processed_actions[0].payload.object_ids) == {"object1", "object2"}

    optimized_actions = service._remove_redundant_visibility_actions(processed_actions)
    assert len(optimized_actions) == len(actions) - 2


def test_complex_visibility_actions_no_merge(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_3"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_4"],
        action_payloads["screenshot"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_5"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_3"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_6"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_4"],
        action_payloads["screenshot"],
    ]
    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions)

    optimized_actions = service._remove_redundant_visibility_actions(processed_actions)
    assert len(optimized_actions) == len(actions)


def test_complex_visibility_actions_no_merge(action_payloads):
    actions = [
        action_payloads["set_visibility_1"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_2"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_3"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_4"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_5"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_1"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_3"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_2"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_6"],
        action_payloads["screenshot"],
        action_payloads["set_visibility_4"],
        action_payloads["set_field"],
        action_payloads["screenshot"],
    ]
    set_field_actinos = [action for action in actions if action.action == "set-field"]

    service = ShutterBatchService()
    processed_actions = service._merge_visibility_actions(actions)
    assert len(processed_actions) == len(actions)

    optimized_actions = service._remove_redundant_visibility_actions(processed_actions)
    assert len(optimized_actions) == len(actions)

    optimized_actions = service._remove_redundant_set_field_actions(optimized_actions)
    assert len(optimized_actions) == len(actions) - len(set_field_actinos) + 1
