import json
import os

import pydantic as pd
import pytest
import unyt as u

import flow360 as fl
from flow360 import log
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.exceptions import Flow360ConfigurationError, Flow360ValueError

log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_from_cloud(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    project.print_project_tree(is_horizontal=True, line_width=15)

    assert project.length_unit == 1 * fl.u.m
    assert isinstance(project._root_asset, fl.Geometry)
    assert len(project.get_surface_mesh_ids()) == 1
    assert len(project.get_volume_mesh_ids()) == 1
    assert len(project.get_case_ids()) == 2

    current_geometry_id = "geo-2877e124-96ff-473d-864b-11eec8648d42"
    current_surface_mesh_id = "sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a"
    current_volume_mesh_id = "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
    current_case_id = "case-84d4604e-f3cd-4c6b-8517-92a80a3346d3"

    assert project.geometry.id == current_geometry_id
    assert project.surface_mesh.id == current_surface_mesh_id
    assert project.volume_mesh.id == current_volume_mesh_id
    assert project.case.id == current_case_id

    assert project.geometry.params
    assert project.surface_mesh.params
    assert project.volume_mesh.params

    for case_id in project.get_case_ids():
        project.project_tree.remove_node(node_id=case_id)
    error_msg = "No Case is available in this project."
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=current_case_id)


def test_root_asset_entity_change_reflection(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    geo = project.geometry
    geo["wing"].private_attribute_color = "red"

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            outputs=[fl.SurfaceOutput(surfaces=geo["*"], output_fields=["Cp"])],
        )
    params = set_up_params_for_uploading(
        params=params,
        root_asset=project._root_asset,
        length_unit=project.length_unit,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )

    assert (
        params.private_attribute_asset_cache.project_entity_info.grouped_faces[0][
            0
        ].private_attribute_color
        == "red"
    )


def test_get_asset_with_id(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    print(f"The correct asset_id: {project.case.id}")

    query_id = "  case-84d4604e-f3cd-4c6b-8517-92a80a3346d3   "
    assert project.get_case(asset_id=query_id)

    query_id = "=case-84d4604e-f3cd-4c6b-8517-92a80a3346d3&"
    assert project.get_case(asset_id=query_id)

    query_id = "case"
    error_msg = (
        r"The supplied ID \(" + query_id + r"\) does not have a proper surffix-ID structure."
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "sm-123456"
    error_msg = r"The input asset ID \(" + query_id + r"\) is not a Case ID."
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "case-b0"
    error_msg = (
        r"The input asset ID \(" + query_id + r"\) is too short to retrieve the correct asset."
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "case-1234567-abcd"
    error_msg = (
        r"This asset does not exist in this project. Please check the input asset ID \("
        + query_id
        + r"\)"
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)


@pytest.mark.usefixtures("s3_download_override")
def test_run(mock_response, capsys):
    capsys.readouterr()
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    parent_case = project.get_case("case-69b8c249")
    case = project.case  # case-84d4604e-f
    params = case.params

    warning_msg = "The VolumeMesh that matches the input already exists in project. No new VolumeMesh will be generated."
    project.generate_volume_mesh(params=params)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text

    warning_msg = (
        "The Case that matches the input already exists in project. No new Case will be generated."
    )
    project.run_case(params=params, fork_from=parent_case)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text

    error_msg = r"Input should be 'SurfaceMesh', 'VolumeMesh' or 'Case'"
    with pytest.raises(pd.ValidationError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Geometry")

    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(Case\) "
        r"cannot be later than 'up_to' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Case")

    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")
    params = project_vm.case.params

    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(SurfaceMesh\) "
        r"must be later than 'source_item_type' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project_vm.run_case(params=params, start_from="SurfaceMesh")


def test_conflicting_entity_grouping_tags(mock_response, capsys):
    with open(
        os.path.join(os.path.dirname(__file__), "data", "simulation_by_face_id.json"), "r"
    ) as f:
        params_as_dict = json.load(f)

    params, _, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level=None,
    )

    assert params.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceId"
    assert params.private_attribute_asset_cache.project_entity_info.edge_group_tag == "edgeId"
    assert params.private_attribute_asset_cache.project_entity_info.body_group_tag == "groupByFile"

    geo = Geometry.from_local_storage(
        geometry_id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "geometry_grouped_by_file"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
                name="TEST",
                cloud_path_prefix="/",
                status="processed",
            )
        ),
    )

    assert geo.face_group_tag == "groupByBodyId"

    geo.internal_registry = geo._entity_info.get_registry(geo.internal_registry)

    new_params = set_up_params_for_uploading(
        geo, 1 * u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    assert new_params.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceId"
    assert new_params.private_attribute_asset_cache.project_entity_info.edge_group_tag == "edgeId"
    assert (
        new_params.private_attribute_asset_cache.project_entity_info.body_group_tag == "groupByFile"
    )

    geo.group_faces_by_tag("allInOne")

    with pytest.raises(Flow360ConfigurationError, match="Conflicting entity grouping tags found"):
        new_params = set_up_params_for_uploading(
            geo, 1 * u.m, params, use_beta_mesher=False, use_geometry_AI=False
        )

    with pytest.raises(Flow360ConfigurationError, match="Conflicting entity grouping tags found"):
        geo.group_faces_by_tag("groupByBodyId")
        params_as_dict[""]
        params, _, _ = validate_model(
            params_as_dict=params_as_dict,
            validated_by=ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level=None,
        )