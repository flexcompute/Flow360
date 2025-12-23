import pytest

from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
)


class TestParamsValidationInfo:
    """Tests for ParamsValidationInfo class."""

    def test_entity_transformation_detected_none(self):
        """Test entity_transformation_detected when no transformation or mirroring exists."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": None,
                "mirror_status": None,
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is False

    def test_entity_transformation_detected_empty(self):
        """Test entity_transformation_detected when status objects are empty."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": {"assignments": []},
                "mirror_status": {"mirrored_geometry_body_groups": [], "mirrored_surfaces": []},
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is False

    def test_entity_transformation_detected_coordinate_system(self):
        """Test detection of coordinate system assignments."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": {"assignments": [{"some": "assignment"}]},
                "mirror_status": None,
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is True

    def test_entity_transformation_detected_mirror_groups(self):
        """Test detection of mirrored geometry groups."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": None,
                "mirror_status": {
                    "mirrored_geometry_body_groups": [{"some": "group"}],
                    "mirrored_surfaces": [],
                },
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is True

    def test_entity_transformation_detected_mirror_surfaces(self):
        """Test detection of mirrored surfaces."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": None,
                "mirror_status": {
                    "mirrored_geometry_body_groups": [],
                    "mirrored_surfaces": [{"some": "surface"}],
                },
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is True

    def test_entity_transformation_detected_both(self):
        """Test detection when both transformations are present."""
        param_dict = {
            "private_attribute_asset_cache": {
                "coordinate_system_status": {"assignments": [{"some": "assignment"}]},
                "mirror_status": {"mirrored_surfaces": [{"some": "surface"}]},
            }
        }
        info = ParamsValidationInfo(param_dict, referenced_expressions=[])
        assert info.entity_transformation_detected is True
