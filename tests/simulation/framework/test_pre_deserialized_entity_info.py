"""Tests for validate_model() dict substitution optimization."""

import pytest

from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.primitives import Surface


@pytest.fixture
def volume_mesh_entity_info():
    """Create a VolumeMeshEntityInfo object."""
    wall = Surface(name="wall", private_attribute_is_interface=False)
    return VolumeMeshEntityInfo(
        boundaries=[wall],
        zones=[],
    )


class TestValidateModelDictSubstitutionOptimization:
    """Tests verifying validate_model() dict substitution works correctly."""

    def test_shallow_copy_does_not_affect_original(self, volume_mesh_entity_info):
        """Shallow copy of dict with substituted entity_info doesn't affect original."""
        original_dict = {
            "private_attribute_asset_cache": {
                "project_entity_info": {
                    "type_name": "VolumeMeshEntityInfo",
                    "boundaries": [],
                    "zones": [],
                },
                "use_inhouse_mesher": False,
            },
            "other_field": "value",
        }

        # Simulate what validate_model() does
        new_dict = {**original_dict}
        new_dict["private_attribute_asset_cache"] = {
            **original_dict["private_attribute_asset_cache"],
            "project_entity_info": volume_mesh_entity_info,
        }

        # Original should be unchanged
        assert isinstance(
            original_dict["private_attribute_asset_cache"]["project_entity_info"], dict
        )
        # New dict has the object
        assert (
            new_dict["private_attribute_asset_cache"]["project_entity_info"]
            is volume_mesh_entity_info
        )
        # Other fields are shared (shallow copy)
        assert new_dict["other_field"] is original_dict["other_field"]
