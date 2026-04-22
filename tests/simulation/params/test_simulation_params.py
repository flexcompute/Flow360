import json
import os
import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.migration.extra_operating_condition import (
    operating_condition_from_mach_muref,
)

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

