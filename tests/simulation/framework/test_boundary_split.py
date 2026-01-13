"""
Tests for boundary_split.py - BoundaryNameLookupTable and related functions.
"""

import pytest

from flow360.component.simulation.framework.boundary_split import (
    BoundaryNameLookupTable,
    SplitType,
    update_entities_in_model,
)
from flow360.component.simulation.primitives import Surface


class TestBoundaryNameLookupTable:
    """Tests for BoundaryNameLookupTable."""

    @pytest.fixture
    def multizone_mesh_metadata(self):
        """Mesh metadata for a multi-zone mesh (e.g., rotating sphere case)."""
        return {
            "zones": {
                "farfieldBlock": {
                    "boundaryNames": [
                        "farfieldBlock/farfield",
                        "farfieldBlock/rotationInterface",
                    ],
                },
                "nearfieldBlock": {
                    "boundaryNames": [
                        "nearfieldBlock/ball",
                        "nearfieldBlock/rotationInterface",
                    ],
                },
            }
        }

    @pytest.fixture
    def single_zone_mesh_metadata(self):
        """Mesh metadata for a single-zone mesh."""
        return {
            "zones": {
                "fluid": {
                    "boundaryNames": [
                        "fluid/wing",
                        "fluid/fuselage",
                        "fluid/farfield",
                    ],
                },
            }
        }

    def test_lookup_by_base_name(self, multizone_mesh_metadata):
        """Test that lookup by base_name works (original behavior)."""
        lookup_table = BoundaryNameLookupTable(multizone_mesh_metadata)

        # Lookup by base name should find the full name
        split_infos = lookup_table.get_split_info("farfield")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "farfieldBlock/farfield"
        assert split_infos[0].split_type == SplitType.ZONE_PREFIX

        split_infos = lookup_table.get_split_info("ball")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "nearfieldBlock/ball"

    def test_lookup_by_full_name_passthrough(self, multizone_mesh_metadata):
        """
        Test that lookup by full_name works (passthrough behavior).

        This is the bug fix case: when entity.name is already a full_name
        (e.g., from volume mesh), the lookup should still succeed.
        """
        lookup_table = BoundaryNameLookupTable(multizone_mesh_metadata)

        # Lookup by full name should also work (passthrough)
        split_infos = lookup_table.get_split_info("farfieldBlock/farfield")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "farfieldBlock/farfield"

        split_infos = lookup_table.get_split_info("nearfieldBlock/ball")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "nearfieldBlock/ball"

    def test_lookup_multizone_same_base_name(self, multizone_mesh_metadata):
        """
        Test lookup when the same base_name appears in multiple zones.

        rotationInterface appears in both farfieldBlock and nearfieldBlock.
        """
        lookup_table = BoundaryNameLookupTable(multizone_mesh_metadata)

        # Base name lookup should return both
        split_infos = lookup_table.get_split_info("rotationInterface")
        assert len(split_infos) == 2

        full_names = {info.full_name for info in split_infos}
        assert full_names == {
            "farfieldBlock/rotationInterface",
            "nearfieldBlock/rotationInterface",
        }

        # At least one should be MULTI_ZONE type
        split_types = {info.split_type for info in split_infos}
        assert SplitType.MULTI_ZONE in split_types

    def test_lookup_nonexistent_name(self, multizone_mesh_metadata):
        """Test that lookup for nonexistent name returns empty list."""
        lookup_table = BoundaryNameLookupTable(multizone_mesh_metadata)

        split_infos = lookup_table.get_split_info("nonexistent")
        assert split_infos == []

    def test_single_zone_lookup(self, single_zone_mesh_metadata):
        """Test lookup in single-zone mesh."""
        lookup_table = BoundaryNameLookupTable(single_zone_mesh_metadata)

        # Base name lookup
        split_infos = lookup_table.get_split_info("wing")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "fluid/wing"

        # Full name lookup (passthrough)
        split_infos = lookup_table.get_split_info("fluid/wing")
        assert len(split_infos) == 1
        assert split_infos[0].full_name == "fluid/wing"

    def test_empty_metadata(self):
        """Test with empty metadata."""
        lookup_table = BoundaryNameLookupTable({})
        assert lookup_table.get_split_info("anything") == []

        lookup_table = BoundaryNameLookupTable({"zones": {}})
        assert lookup_table.get_split_info("anything") == []

        lookup_table = BoundaryNameLookupTable({"zones": None})
        assert lookup_table.get_split_info("anything") == []
