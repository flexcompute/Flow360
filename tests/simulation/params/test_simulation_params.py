import json
import os
import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.migration.extra_operating_condition import (
    operating_condition_from_mach_muref,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    _inject_body_group_transformations_for_mesher,
)
from flow360.component.simulation.unit_system import SI_unit_system

assertions = unittest.TestCase("__init__")


def test_mach_muref_op_cond():
    condition = operating_condition_from_mach_muref(
        mach=0.2,
        mu_ref=4e-8,
        temperature=288.15 * u.K,
        alpha=2.0 * u.deg,
        beta=0.0 * u.deg,
        project_length_unit=u.m,
    )
    assertions.assertAlmostEqual(condition.thermal_state.dynamic_viscosity.value, 1.78929763e-5)
    assertions.assertAlmostEqual(condition.thermal_state.density.value, 1.31452332)
    assertions.assertAlmostEqual(
        condition.flow360_reynolds_number(length_unit=1 * u.m),
        (1.0 / 4e-8) * condition.mach,
    )

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        operating_condition_from_mach_muref(
            mach=0.2,
            mu_ref=0,
            temperature=288.15 * u.K,
        )


def test_geometry_entity_info_to_file_list_and_entity_to_file_map():
    simulation_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "geometry_metadata_asset_cache_mixed_file.json",
    )
    with open(simulation_path, "r") as file:
        geometry_entity_info_dict = json.load(file)
        geometry_entity_info = GeometryEntityInfo.model_validate(geometry_entity_info_dict)

    assert geometry_entity_info._get_processed_file_list() == (
        ["airplane_simple_obtained_from_csm_by_esp.step.egads"],
        ["airplane_translate_in_z_-5.stl", "farfield_only_sphere_volume_mesh.lb8.ugrid"],
    )

    assert sorted(
        geometry_entity_info._get_id_to_file_map(entity_type_name="body").items()
    ) == sorted(
        {
            "body00001": "airplane_simple_obtained_from_csm_by_esp.step.egads",
            "airplane_translate_in_z_-5.stl": "airplane_translate_in_z_-5.stl",
            "farfield_only_sphere_volume_mesh.lb8.ugrid": "farfield_only_sphere_volume_mesh.lb8.ugrid",
        }.items()
    )


def test_transformation_matrix(mock_geometry):
    def _get_selected_grouped_bodies(entity_info_dict: dict) -> list[dict]:
        grouped_bodies = entity_info_dict.get("grouped_bodies", None)
        assert isinstance(grouped_bodies, list)

        body_group_tag = entity_info_dict.get("body_group_tag")
        body_attribute_names = entity_info_dict.get("body_attribute_names")
        assert isinstance(body_group_tag, str)
        assert isinstance(body_attribute_names, list)
        assert body_group_tag in body_attribute_names

        selected_group_index = body_attribute_names.index(body_group_tag)
        selected_group = grouped_bodies[selected_group_index]
        assert isinstance(selected_group, list)
        assert selected_group
        return selected_group

    entity_info = mock_geometry.entity_info
    candidate_body_group_tag = None
    if getattr(entity_info, "body_attribute_names", None):
        for tag in ("groupByBodyId", "bodyId"):
            if tag in entity_info.body_attribute_names:
                candidate_body_group_tag = tag
                break
    if candidate_body_group_tag is not None:
        entity_info._force_set_attr("body_group_tag", candidate_body_group_tag)

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        target_body_group = body_groups[0]

        with SI_unit_system:
            cs_parent = CoordinateSystem(name="parent", translation=[10, 0, 0] * u.m)
            cs_child = CoordinateSystem(name="child", translation=[0, 5, 0] * u.m)

        cs_parent = draft.coordinate_systems.add(coordinate_system=cs_parent)
        cs_child = draft.coordinate_systems.add(coordinate_system=cs_child, parent=cs_parent)
        draft.coordinate_systems.assign(entities=target_body_group, coordinate_system=cs_child)

        with SI_unit_system:
            params = SimulationParams(operating_condition=AerospaceCondition())

        processed_params = set_up_params_for_uploading(
            root_asset=mock_geometry,
            length_unit=1 * u.m,
            params=params,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

    json_data = processed_params.model_dump(mode="json", exclude_none=True)
    _inject_body_group_transformations_for_mesher(
        json_data=json_data,
        input_params=processed_params,
        mesh_unit=1 * u.m,
    )

    entity_info_dict = json_data["private_attribute_asset_cache"]["project_entity_info"]
    selected_group = _get_selected_grouped_bodies(entity_info_dict=entity_info_dict)

    expected_child_matrix = [
        1.0,
        0.0,
        0.0,
        10.0,
        0.0,
        1.0,
        0.0,
        5.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    target_body_group_dict = None
    for body_group_dict in selected_group:
        if (
            body_group_dict.get("private_attribute_entity_type_name")
            == target_body_group.private_attribute_entity_type_name
            and body_group_dict.get("private_attribute_id")
            == target_body_group.private_attribute_id
        ):
            target_body_group_dict = body_group_dict

    assert isinstance(target_body_group_dict, dict)

    for body_group_dict in selected_group:
        transformation = body_group_dict.get("transformation")
        assert isinstance(transformation, dict)
        assert list(transformation.keys()) == ["private_attribute_matrix"], (
            f"Expected transformation to only contain 'private_attribute_matrix', "
            f"but got keys: {list(transformation.keys())}"
        )
        assert isinstance(transformation.get("private_attribute_matrix"), list)
        assert len(transformation["private_attribute_matrix"]) == 12

    assert (
        target_body_group_dict["transformation"]["private_attribute_matrix"]
        == expected_child_matrix
    )

    for body_group_dict in selected_group:
        if body_group_dict is target_body_group_dict:
            continue
        assert body_group_dict["transformation"]["private_attribute_matrix"] == [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
