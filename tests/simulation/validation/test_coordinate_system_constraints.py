"""
Tests for _check_coordinate_system_constraints validation function.
"""

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemStatus,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.validation.validation_simulation_params import (
    _check_coordinate_system_constraints,
)


class MockParamsValidationInfo:
    """Mock ParamsValidationInfo for testing."""

    def __init__(self, use_geometry_AI: bool = False):
        self.use_geometry_AI = use_geometry_AI


class MockParams:
    """Mock SimulationParams for testing."""

    def __init__(self, asset_cache: AssetCache):
        self.private_attribute_asset_cache = asset_cache


# ============================================================================
# Test Cases - GAI Requirement for GeometryBodyGroup
# ============================================================================


def test_geometry_body_group_without_gai_raises():
    """GeometryBodyGroup + no GAI -> should raise."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(
                        entity_type="GeometryBodyGroup", entity_id="body-group-id"
                    )
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError, match="Coordinate system assignment to GeometryBodyGroup"):
        _check_coordinate_system_constraints(params, param_info)


def test_geometry_body_group_with_gai_passes():
    """GeometryBodyGroup + GAI enabled -> should NOT raise."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(
                        entity_type="GeometryBodyGroup", entity_id="body-group-id"
                    )
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=True,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=True)

    # Should not raise
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


def test_box_without_gai_passes():
    """Box assigned to coordinate system + no GAI -> should NOT raise (GAI not required)."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise - Box doesn't require GAI
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


def test_cylinder_without_gai_passes():
    """Cylinder assigned to coordinate system + no GAI -> should NOT raise."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(entity_type="Cylinder", entity_id="cylinder-id")
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise - Cylinder doesn't require GAI
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


def test_surface_without_gai_passes():
    """Surface assigned to coordinate system + no GAI -> should NOT raise."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Surface", entity_id="surface-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise - Surface doesn't require GAI
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


# ============================================================================
# Test Cases - Uniform Scaling Validation
# ============================================================================


def test_box_with_non_uniform_scale_raises():
    """Box assigned to non-uniform scale coordinate system -> should raise early."""
    # Create coordinate system with non-uniform scaling (2, 3, 4)
    cs = CoordinateSystem(name="non_uniform_cs", scale=(2.0, 3.0, 4.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError, match="non-uniform scaling"):
        _check_coordinate_system_constraints(params, param_info)


def test_cylinder_with_non_uniform_scale_raises():
    """Cylinder assigned to non-uniform scale coordinate system -> should raise early."""
    cs = CoordinateSystem(name="non_uniform_cs", scale=(1.0, 2.0, 1.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(entity_type="Cylinder", entity_id="cylinder-id")
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError, match="non-uniform scaling"):
        _check_coordinate_system_constraints(params, param_info)


def test_axisymmetric_body_with_non_uniform_scale_raises():
    """AxisymmetricBody assigned to non-uniform scale coordinate system -> should raise early."""
    cs = CoordinateSystem(name="non_uniform_cs", scale=(1.5, 1.5, 2.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(entity_type="AxisymmetricBody", entity_id="axisym-id")
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError, match="non-uniform scaling"):
        _check_coordinate_system_constraints(params, param_info)


def test_surface_with_non_uniform_scale_passes():
    """Surface assigned to non-uniform scale coordinate system -> should NOT raise."""
    cs = CoordinateSystem(name="non_uniform_cs", scale=(2.0, 3.0, 4.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Surface", entity_id="surface-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise - Surface doesn't require uniform scaling
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


def test_box_with_uniform_scale_passes():
    """Box assigned to uniform scale coordinate system -> should NOT raise."""
    cs = CoordinateSystem(name="uniform_cs", scale=(2.0, 2.0, 2.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise - uniform scaling is OK
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


# ============================================================================
# Test Cases - Hierarchical Coordinate Systems with Non-Uniform Scale
# ============================================================================


def test_box_with_parent_non_uniform_scale_raises():
    """Box assigned to child coordinate system where parent has non-uniform scale -> should raise."""
    from flow360.component.simulation.draft_context.coordinate_system_manager import (
        CoordinateSystemParent,
    )

    # Parent with non-uniform scale
    parent_cs = CoordinateSystem(name="parent_cs", scale=(1.0, 2.0, 1.0))
    # Child with uniform scale (but composed matrix will be non-uniform)
    child_cs = CoordinateSystem(name="child_cs", scale=(1.0, 1.0, 1.0))

    cs_status = CoordinateSystemStatus(
        coordinate_systems=[parent_cs, child_cs],
        parents=[
            CoordinateSystemParent(
                coordinate_system_id=parent_cs.private_attribute_id, parent_id=None
            ),
            CoordinateSystemParent(
                coordinate_system_id=child_cs.private_attribute_id,
                parent_id=parent_cs.private_attribute_id,
            ),
        ],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=child_cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should raise because composed matrix has non-uniform scaling
    with pytest.raises(ValueError, match="non-uniform scaling"):
        _check_coordinate_system_constraints(params, param_info)


# ============================================================================
# Test Cases - No Coordinate System Usage
# ============================================================================


def test_no_coordinate_system_status_passes():
    """No coordinate system status -> should pass."""
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=None,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


def test_empty_assignments_passes():
    """Empty assignments -> should pass."""
    cs = CoordinateSystem(name="test_cs")
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[],  # Empty assignments
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    # Should not raise
    result = _check_coordinate_system_constraints(params, param_info)
    assert result is params


# ============================================================================
# Test Cases - Error Message Quality
# ============================================================================


def test_error_message_includes_coordinate_system_name():
    """Error message should include the coordinate system name."""
    cs = CoordinateSystem(name="my_special_cs", scale=(1.0, 2.0, 3.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id")],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError, match="my_special_cs"):
        _check_coordinate_system_constraints(params, param_info)


def test_error_message_includes_entity_types():
    """Error message should include the entity types that require uniform scaling."""
    cs = CoordinateSystem(name="non_uniform_cs", scale=(1.0, 2.0, 3.0))
    cs_status = CoordinateSystemStatus(
        coordinate_systems=[cs],
        parents=[],
        assignments=[
            CoordinateSystemAssignmentGroup(
                coordinate_system_id=cs.private_attribute_id,
                entities=[
                    CoordinateSystemEntityRef(entity_type="Box", entity_id="box-id"),
                    CoordinateSystemEntityRef(entity_type="Cylinder", entity_id="cylinder-id"),
                ],
            )
        ],
    )
    asset_cache = AssetCache(
        use_inhouse_mesher=False,
        use_geometry_AI=False,
        coordinate_system_status=cs_status,
    )
    params = MockParams(asset_cache)
    param_info = MockParamsValidationInfo(use_geometry_AI=False)

    with pytest.raises(ValueError) as exc_info:
        _check_coordinate_system_constraints(params, param_info)

    # Error should mention both Box and Cylinder
    error_msg = str(exc_info.value)
    assert "Box" in error_msg
    assert "Cylinder" in error_msg
