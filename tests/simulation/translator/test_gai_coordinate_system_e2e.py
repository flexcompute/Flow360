"""End-to-end test for coordinate system transformation in GAI surface meshing translation.

This test verifies that when a user creates a draft, assigns a coordinate system to body groups,
the transformation matrix is properly injected into the translated GAI JSON.
"""

import json
import os

import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def geometry_with_multiple_body_groups():
    """Fixture for geometry with multiple body groups (cube-holes.egads and cylinder.stl)."""
    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="multi_body_group_geometry",
                cloud_path_prefix="multi_body",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")
    return geometry


def test_coordinate_system_transformation_in_gai_translated_json(mock_geometry):
    """Test that coordinate system transformation matrix is injected into GAI translated JSON.

    This end-to-end test verifies:
    1. A draft can be created from geometry
    2. A coordinate system with non-trivial transformation can be assigned to body groups
    3. The transformation matrix is properly injected into the translated GAI JSON
    4. The matrix values are correct (matching the expected transformation)
    """
    # Set up the registry for the mock geometry
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        # Get available body groups
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0, "Expected at least one body group in mock geometry"
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Create a coordinate system with a non-trivial transformation:
            # - Translation of (10, 20, 30) meters
            # - 90 degree rotation around Z axis
            coordinate_system = CoordinateSystem(
                name="test_vehicle_frame",
                reference_point=[0, 0, 0] * u.m,
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=90 * u.deg,
                scale=(1.0, 1.0, 1.0),
                translation=[10, 20, 30] * u.m,
            )

            # Assign the coordinate system to the body group
            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=coordinate_system,
            )

            # Verify assignment was made
            assigned_cs = draft.coordinate_systems._get_coordinate_system_for_entity(
                entity=target_body_group
            )
            assert assigned_cs is not None
            assert assigned_cs.name == "test_vehicle_frame"

            # Create SimulationParams with GAI-compatible settings
            farfield = AutomatedFarfield()
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,  # GAI-specific setting
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[farfield],
                ),
                operating_condition=AerospaceCondition(
                    velocity_magnitude=10 * u.m / u.s,
                ),
            )

            # Set up params for uploading - this populates asset cache with coordinate system status
            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,  # Enable GAI mode
            )

    # Verify coordinate system status was stored in asset cache
    assert processed_params.private_attribute_asset_cache.coordinate_system_status is not None
    cs_status = processed_params.private_attribute_asset_cache.coordinate_system_status
    assert len(cs_status.coordinate_systems) == 1
    assert cs_status.coordinate_systems[0].name == "test_vehicle_frame"

    # Call the GAI translator
    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    # Verify the JSON is serializable
    json_str = json.dumps(translated)
    assert json_str is not None

    # Verify the structure of the translated JSON for GAI mode
    assert "private_attribute_asset_cache" in translated
    assert "project_entity_info" in translated["private_attribute_asset_cache"]
    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    assert "grouped_bodies" in entity_info

    # Find the body group in the translated JSON and verify the transformation matrix
    grouped_bodies = entity_info["grouped_bodies"]
    found_transformation = False
    transformation_matrix = None

    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                assert "transformation" in body_group
                transformation = body_group["transformation"]

                # Verify transformation only contains private_attribute_matrix (no redundant fields)
                assert list(transformation.keys()) == ["private_attribute_matrix"], (
                    f"Expected transformation to only contain 'private_attribute_matrix', "
                    f"but got keys: {list(transformation.keys())}"
                )

                transformation_matrix = transformation["private_attribute_matrix"]
                found_transformation = True
                break
        if found_transformation:
            break

    assert found_transformation, (
        f"Expected to find transformation for body group '{target_body_group.name}' "
        f"(id: {target_body_group.private_attribute_id})"
    )

    # Verify the matrix is a 12-element list (3x4 matrix in row-major order)
    assert isinstance(transformation_matrix, list)
    assert len(transformation_matrix) == 12

    # Calculate expected matrix for 90 degree rotation around Z axis + translation
    # Rotation matrix for 90 degrees around Z:
    # [ cos(90)  -sin(90)  0 ]   [ 0  -1  0 ]
    # [ sin(90)   cos(90)  0 ] = [ 1   0  0 ]
    # [   0         0      1 ]   [ 0   0  1 ]
    # Plus translation [10, 20, 30]
    expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    expected_translation = np.array([10, 20, 30], dtype=np.float64)
    expected_matrix = np.hstack([expected_rotation, expected_translation[:, np.newaxis]])
    expected_flat = expected_matrix.flatten().tolist()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_flat,
        atol=1e-10,
        err_msg="Transformation matrix does not match expected rotation + translation",
    )


def test_coordinate_system_with_parent_hierarchy_in_gai_json(mock_geometry):
    """Test that coordinate system with parent hierarchy produces composed transformation matrix.

    This test verifies:
    1. Parent-child coordinate system hierarchy can be created
    2. The composed transformation matrix is correctly computed
    3. The composed matrix is injected into the translated GAI JSON
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Create parent coordinate system with translation
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="vehicle_frame",
                    translation=(100, 0, 0) * u.m,
                )
            )

            # Create child coordinate system with additional translation
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="component_frame",
                    translation=(0, 50, 0) * u.m,
                ),
                parent=parent_cs,
            )

            # Assign child coordinate system to body group
            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    # Call the GAI translator
    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    # Find the transformation matrix in the translated JSON
    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Expected composed transformation:
    # parent: translate (100, 0, 0)
    # child: translate (0, 50, 0)
    # Composed: parent rotation * child_translation + parent_translation
    # = identity * (0, 50, 0) + (100, 0, 0) = (100, 50, 0)
    expected_matrix = np.array(
        [[1, 0, 0, 100], [0, 1, 0, 50], [0, 0, 1, 0]], dtype=np.float64
    ).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Composed transformation matrix does not match expected parent-child composition",
    )


def test_deeply_nested_coordinate_systems_three_levels(mock_geometry):
    """Test coordinate systems with 3-level deep nesting (grandparent -> parent -> child).

    This test verifies:
    1. Three-level coordinate system hierarchy can be created
    2. The composed transformation matrix correctly applies all three levels
    3. Translation composition works correctly through the hierarchy
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Create grandparent coordinate system (world -> vehicle frame)
            grandparent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="world_frame",
                    translation=(1000, 0, 0) * u.m,
                )
            )

            # Create parent coordinate system (vehicle -> component frame)
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="vehicle_frame",
                    translation=(0, 200, 0) * u.m,
                ),
                parent=grandparent_cs,
            )

            # Create child coordinate system (component -> sub-component frame)
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="component_frame",
                    translation=(0, 0, 30) * u.m,
                ),
                parent=parent_cs,
            )

            # Assign the deepest child coordinate system to the body group
            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    # Call the GAI translator
    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    # Find the transformation matrix
    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Expected composed transformation:
    # grandparent: translate (1000, 0, 0)
    # parent: translate (0, 200, 0)
    # child: translate (0, 0, 30)
    # All have identity rotation, so composed translation = (1000, 200, 30)
    expected_matrix = np.array(
        [[1, 0, 0, 1000], [0, 1, 0, 200], [0, 0, 1, 30]], dtype=np.float64
    ).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="3-level nested transformation matrix does not match expected composition",
    )


def test_nested_coordinate_systems_with_rotation_and_translation(mock_geometry):
    """Test nested coordinate systems where parent has rotation and child has translation.

    This test verifies that rotation in parent coordinate system correctly
    transforms the child's translation vector.

    Scenario:
    - Parent: 90 degree rotation around Z axis
    - Child: Translation (10, 0, 0) in parent's frame

    Expected result:
    - Child's translation (10, 0, 0) rotated by parent's 90-deg Z rotation becomes (0, 10, 0)
    - Combined rotation is 90 deg around Z
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Parent: 90 degree rotation around Z axis
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="rotated_frame",
                    axis_of_rotation=(0, 0, 1),
                    angle_of_rotation=90 * u.deg,
                )
            )

            # Child: translation in local X direction
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="translated_frame",
                    translation=(10, 0, 0) * u.m,
                ),
                parent=parent_cs,
            )

            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Expected:
    # Parent rotation (90 deg around Z): [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # Child has identity rotation and translation (10, 0, 0)
    # Composed rotation = parent_rotation * child_rotation = parent_rotation
    # Composed translation = parent_rotation * child_translation + parent_translation
    #                      = [[0,-1,0],[1,0,0],[0,0,1]] * [10,0,0] + [0,0,0]
    #                      = [0, 10, 0]
    expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    expected_translation = np.array([0, 10, 0], dtype=np.float64)
    expected_matrix = np.hstack([expected_rotation, expected_translation[:, np.newaxis]]).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Rotation + translation composition does not match expected result",
    )


def test_nested_coordinate_systems_with_rotation_chain(mock_geometry):
    """Test nested coordinate systems with rotations at multiple levels.

    This test verifies that rotations compose correctly through the hierarchy.

    Scenario:
    - Grandparent: 90 degree rotation around Z axis
    - Parent: 90 degree rotation around Y axis (in grandparent's frame)
    - Child: 90 degree rotation around X axis (in parent's frame)

    This creates a complex combined rotation that tests matrix multiplication order.
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Grandparent: 90 degree rotation around Z
            grandparent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="z_rotated_frame",
                    axis_of_rotation=(0, 0, 1),
                    angle_of_rotation=90 * u.deg,
                )
            )

            # Parent: 90 degree rotation around Y
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="y_rotated_frame",
                    axis_of_rotation=(0, 1, 0),
                    angle_of_rotation=90 * u.deg,
                ),
                parent=grandparent_cs,
            )

            # Child: 90 degree rotation around X
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="x_rotated_frame",
                    axis_of_rotation=(1, 0, 0),
                    angle_of_rotation=90 * u.deg,
                ),
                parent=parent_cs,
            )

            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Calculate expected rotation manually:
    # Rz(90): [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # Ry(90): [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    # Rx(90): [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    # Combined = Rz @ Ry @ Rx
    rotation_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    rotation_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    rotation_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    combined_rotation = rotation_z @ rotation_y @ rotation_x
    expected_matrix = np.hstack([combined_rotation, np.array([[0], [0], [0]])]).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Chained rotation composition does not match expected result",
    )


def test_nested_coordinate_systems_with_scale_and_rotation(mock_geometry):
    """Test nested coordinate systems with scale at parent and rotation at child.

    This test verifies that scale and rotation compose correctly.

    Scenario:
    - Parent: Scale by (2, 2, 2)
    - Child: 90 degree rotation around Z axis + translation (5, 0, 0)

    Note: Scale is applied first (at origin), then rotation, then translation.
    The composition should apply parent's scale, then child's transform.
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Parent: uniform scale of 2x
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="scaled_frame",
                    scale=(2, 2, 2),
                )
            )

            # Child: rotation + translation
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="rotated_translated_frame",
                    axis_of_rotation=(0, 0, 1),
                    angle_of_rotation=90 * u.deg,
                    translation=(5, 0, 0) * u.m,
                ),
                parent=parent_cs,
            )

            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Parent: scale(2,2,2) -> rotation part is [[2,0,0],[0,2,0],[0,0,2]], translation [0,0,0]
    # Child: rotation Rz(90) + translation (5,0,0)
    #   -> child rotation: [[0,-1,0],[1,0,0],[0,0,1]], translation [5,0,0]
    # Combined rotation = parent_rot @ child_rot = [[2,0,0],[0,2,0],[0,0,2]] @ [[0,-1,0],[1,0,0],[0,0,1]]
    #                   = [[0,-2,0],[2,0,0],[0,0,2]]
    # Combined translation = parent_rot @ child_trans + parent_trans
    #                      = [[2,0,0],[0,2,0],[0,0,2]] @ [5,0,0] + [0,0,0]
    #                      = [10, 0, 0]
    parent_rotation = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
    child_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    child_translation = np.array([5, 0, 0], dtype=np.float64)

    combined_rotation = parent_rotation @ child_rotation
    combined_translation = parent_rotation @ child_translation

    expected_matrix = np.hstack([combined_rotation, combined_translation[:, np.newaxis]]).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Scale + rotation composition does not match expected result",
    )


def test_complex_nested_coordinate_system_all_transforms(mock_geometry):
    """Test complex nested coordinate system with all transformation types at each level.

    This comprehensive test verifies:
    - 3-level hierarchy
    - Each level has rotation, scale, and translation
    - All transformations compose correctly

    Scenario:
    - Grandparent: origin offset, 45 deg rotation around Z, scale 1.5x
    - Parent: translation (10, 0, 0)
    - Child: 90 deg rotation around X, translation (0, 5, 0)
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) > 0
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Grandparent: 45 degree rotation around Z with 1.5x scale
            grandparent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="world_frame",
                    axis_of_rotation=(0, 0, 1),
                    angle_of_rotation=45 * u.deg,
                    scale=(1.5, 1.5, 1.5),
                )
            )

            # Parent: just translation
            parent_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="vehicle_frame",
                    translation=(10, 0, 0) * u.m,
                ),
                parent=grandparent_cs,
            )

            # Child: rotation around X + translation
            child_cs = draft.coordinate_systems.add(
                coordinate_system=CoordinateSystem(
                    name="component_frame",
                    axis_of_rotation=(1, 0, 0),
                    angle_of_rotation=90 * u.deg,
                    translation=(0, 5, 0) * u.m,
                ),
                parent=parent_cs,
            )

            draft.coordinate_systems.assign(
                entities=target_body_group,
                coordinate_system=child_cs,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    transformation_matrix = None
    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            if body_group.get("private_attribute_id") == target_body_group.private_attribute_id:
                transformation_matrix = body_group["transformation"]["private_attribute_matrix"]
                break
        if transformation_matrix is not None:
            break

    assert transformation_matrix is not None

    # Calculate expected matrix step by step
    cos45 = np.cos(np.deg2rad(45))
    sin45 = np.sin(np.deg2rad(45))
    scale = 1.5

    # Grandparent: Rz(45) * scale(1.5)
    grandparent_rot = scale * np.array(
        [[cos45, -sin45, 0], [sin45, cos45, 0], [0, 0, 1]], dtype=np.float64
    )
    grandparent_trans = np.array([0, 0, 0], dtype=np.float64)

    # Parent: identity rotation, translation (10, 0, 0)
    parent_rot = np.eye(3, dtype=np.float64)
    parent_trans = np.array([10, 0, 0], dtype=np.float64)

    # Child: Rx(90), translation (0, 5, 0)
    child_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    child_trans = np.array([0, 5, 0], dtype=np.float64)

    # Compose grandparent and parent
    # combined_rot = grandparent_rot @ parent_rot
    # combined_trans = grandparent_rot @ parent_trans + grandparent_trans
    gp_p_rot = grandparent_rot @ parent_rot
    gp_p_trans = grandparent_rot @ parent_trans + grandparent_trans

    # Compose (grandparent+parent) and child
    final_rot = gp_p_rot @ child_rot
    final_trans = gp_p_rot @ child_trans + gp_p_trans

    expected_matrix = np.hstack([final_rot, final_trans[:, np.newaxis]]).flatten()

    np.testing.assert_allclose(
        transformation_matrix,
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Complex 3-level nested transformation does not match expected result",
    )


def test_unassigned_body_group_gets_identity_matrix_in_gai_json(mock_geometry):
    """Test that body groups without coordinate system assignment get identity matrix.

    This test verifies:
    1. Body groups not assigned a coordinate system get identity transformation
    2. The identity matrix is properly formatted in the translated JSON
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        # Don't assign any coordinate system - body groups should get identity matrix

        with SI_unit_system:
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    # Find body groups in translated JSON
    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    # Check that unassigned body groups get identity matrix
    identity_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            transformation = body_group.get("transformation", {})
            matrix = transformation.get("private_attribute_matrix")
            if matrix is not None:
                # If matrix is present, verify it's identity for unassigned groups
                np.testing.assert_allclose(
                    matrix,
                    identity_matrix,
                    atol=1e-10,
                    err_msg=f"Expected identity matrix for unassigned body group '{body_group.get('name')}'",
                )


def test_multiple_body_groups_with_different_coordinate_systems(geometry_with_multiple_body_groups):
    """Test assigning different coordinate systems to different body groups.

    This test verifies:
    1. Different body groups can have different coordinate system assignments
    2. Each body group gets its correct transformation matrix
    """
    geometry = geometry_with_multiple_body_groups

    with create_draft(new_run_from=geometry) as draft:
        body_groups = list(draft.body_groups)
        assert len(body_groups) >= 2, "Test requires at least 2 body groups"

        body_group_1 = body_groups[0]
        body_group_2 = body_groups[1]

        with SI_unit_system:
            # Create two different coordinate systems
            cs_1 = CoordinateSystem(
                name="frame_1",
                translation=(10, 0, 0) * u.m,
            )
            cs_2 = CoordinateSystem(
                name="frame_2",
                translation=(0, 20, 0) * u.m,
            )

            draft.coordinate_systems.assign(entities=body_group_1, coordinate_system=cs_1)
            draft.coordinate_systems.assign(entities=body_group_2, coordinate_system=cs_2)

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[AutomatedFarfield()],
                ),
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
            )

            processed_params = set_up_params_for_uploading(
                root_asset=geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=True,
            )

    translated = get_surface_meshing_json(processed_params, mesh_unit=1 * u.m)

    entity_info = translated["private_attribute_asset_cache"]["project_entity_info"]
    grouped_bodies = entity_info["grouped_bodies"]

    # Expected matrices
    expected_matrix_1 = np.array(
        [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64
    ).flatten()
    expected_matrix_2 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 20], [0, 0, 1, 0]], dtype=np.float64
    ).flatten()

    found_1 = False
    found_2 = False

    for body_group_list in grouped_bodies:
        for body_group in body_group_list:
            matrix = body_group.get("transformation", {}).get("private_attribute_matrix")
            if matrix is None:
                continue

            if body_group.get("private_attribute_id") == body_group_1.private_attribute_id:
                np.testing.assert_allclose(matrix, expected_matrix_1.tolist(), atol=1e-10)
                found_1 = True
            elif body_group.get("private_attribute_id") == body_group_2.private_attribute_id:
                np.testing.assert_allclose(matrix, expected_matrix_2.tolist(), atol=1e-10)
                found_2 = True

    assert (
        found_1
    ), f"Did not find transformation for body_group_1 (id: {body_group_1.private_attribute_id})"
    assert (
        found_2
    ), f"Did not find transformation for body_group_2 (id: {body_group_2.private_attribute_id})"
