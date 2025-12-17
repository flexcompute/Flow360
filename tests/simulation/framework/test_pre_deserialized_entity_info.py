"""Tests for pre-deserialized entity_info optimization in validate_model()."""

import pytest

from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.primitives import Surface


@pytest.fixture
def volume_mesh_entity_info():
    """Create a VolumeMeshEntityInfo object."""
    wall = Surface(name="wall", private_attribute_is_interface=False)
    return VolumeMeshEntityInfo(
        boundaries=[wall],
        zones=[],
    )


@pytest.fixture
def volume_mesh_entity_info_dict(volume_mesh_entity_info):
    """Sample VolumeMeshEntityInfo as dict."""
    return volume_mesh_entity_info.model_dump(mode="json")


class TestAssetCacheWithPreDeserializedEntityInfo:
    """Tests for AssetCache accepting pre-deserialized entity_info directly."""

    def test_asset_cache_accepts_entity_info_dict(self, volume_mesh_entity_info_dict):
        """AssetCache correctly deserializes entity_info from dict."""
        asset_cache = AssetCache(project_entity_info=volume_mesh_entity_info_dict)

        assert asset_cache.project_entity_info is not None
        assert isinstance(asset_cache.project_entity_info, VolumeMeshEntityInfo)
        assert len(asset_cache.project_entity_info.boundaries) == 1
        assert asset_cache.project_entity_info.boundaries[0].name == "wall"

    def test_asset_cache_accepts_entity_info_object_directly(self, volume_mesh_entity_info):
        """AssetCache accepts pre-deserialized entity_info object directly."""
        # This is the key optimization: pass the object directly instead of dict
        asset_cache = AssetCache(project_entity_info=volume_mesh_entity_info)

        # Should be the exact same object (identity preserved)
        assert asset_cache.project_entity_info is volume_mesh_entity_info

    def test_object_identity_preserved_with_marker(self, volume_mesh_entity_info):
        """Verify object identity is preserved by checking a marker attribute."""
        # Add a marker attribute to verify identity
        object.__setattr__(volume_mesh_entity_info, "_test_marker", "unique_marker_12345")

        asset_cache = AssetCache(project_entity_info=volume_mesh_entity_info)

        # The marker should still be present (same object)
        assert hasattr(asset_cache.project_entity_info, "_test_marker")
        assert asset_cache.project_entity_info._test_marker == "unique_marker_12345"

    def test_none_entity_info_stays_none(self):
        """None entity_info should remain None."""
        asset_cache = AssetCache(project_entity_info=None)
        assert asset_cache.project_entity_info is None

    def test_full_asset_cache_with_pre_deserialized(self, volume_mesh_entity_info):
        """Full AssetCache with multiple fields and pre-deserialized entity_info."""
        asset_cache = AssetCache(
            project_length_unit={"value": 1.0, "units": "m"},
            project_entity_info=volume_mesh_entity_info,
            use_inhouse_mesher=True,
            use_geometry_AI=False,
        )

        # Verify all fields are correct
        assert asset_cache.project_entity_info is volume_mesh_entity_info
        assert asset_cache.use_inhouse_mesher is True
        assert asset_cache.use_geometry_AI is False
        assert asset_cache.project_length_unit is not None


class TestDictSubstitutionOptimization:
    """Tests verifying the dict substitution approach works correctly."""

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

    def test_different_deserializations_create_distinct_objects(self, volume_mesh_entity_info_dict):
        """Without optimization, each deserialization creates distinct objects."""
        asset_cache1 = AssetCache(project_entity_info=volume_mesh_entity_info_dict)
        asset_cache2 = AssetCache(project_entity_info=volume_mesh_entity_info_dict)

        # Without optimization, each call creates a new entity_info object
        assert asset_cache1.project_entity_info is not asset_cache2.project_entity_info

    def test_same_object_reused_when_passed_directly(self, volume_mesh_entity_info):
        """When same object is passed directly, identity is preserved."""
        asset_cache1 = AssetCache(project_entity_info=volume_mesh_entity_info)
        asset_cache2 = AssetCache(project_entity_info=volume_mesh_entity_info)

        # Both use the same object
        assert asset_cache1.project_entity_info is volume_mesh_entity_info
        assert asset_cache2.project_entity_info is volume_mesh_entity_info
        assert asset_cache1.project_entity_info is asset_cache2.project_entity_info
