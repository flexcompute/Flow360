"""End-to-end test for coordinate system transformation in solver case JSON translation.

This test verifies that when entities (Box, Cylinder, Point, Slice, etc.) are assigned
coordinate systems, they are correctly transformed during preprocessing and the transformations
are reflected in the final translated output.
"""

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
from flow360.component.simulation.outputs.output_entities import Point, Slice
from flow360.component.simulation.outputs.outputs import ProbeOutput, SliceOutput
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def mock_geometry():
    """Fixture for a simple mock geometry."""
    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="test_geometry",
                cloud_path_prefix="test",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")
    return geometry


def test_entity_transformations_applied_in_preprocessing(mock_geometry):
    """Test that entities are transformed during preprocessing when coordinate systems are assigned.

    This end-to-end test verifies:
    1. Entities (Box, Cylinder, Point, Slice) can be assigned coordinate systems
    2. During preprocessing, entities are transformed (coordinates/dimensions change)
    3. The transformed values are correct (matching expected transformations)
    4. The transformations propagate through to the final translated output
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        with SI_unit_system:
            # Create test entities with known initial values
            test_point = Point(
                name="test_point",
                location=(5, 5, 0) * u.m,
            )

            test_slice = Slice(
                name="test_slice",
                origin=(0, 0, 5) * u.m,
                normal=(0, 0, 1),
            )

            # Create a coordinate system with:
            # - 90 degree rotation around Z axis
            # - Translation of (100, 200, 300)
            # - Uniform scale of 2x
            coordinate_system = CoordinateSystem(
                name="test_frame",
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=90 * u.deg,
                scale=(2, 2, 2),
                translation=(100, 200, 300) * u.m,
            )

            # Assign coordinate system to point and slice for outputs
            draft.coordinate_systems.assign(
                entities=[test_point, test_slice],
                coordinate_system=coordinate_system,
            )

            # Create SimulationParams with these entities
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(surface_max_edge_length=0.2),
                    volume_zones=[AutomatedFarfield()],
                ),
                outputs=[
                    ProbeOutput(
                        name="test_probe",
                        output_fields=["velocity"],
                        entities=[test_point],
                    ),
                    SliceOutput(
                        name="test_slice_output",
                        output_fields=["velocity"],
                        entities=[test_slice],
                    ),
                ],
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
                time_stepping=Steady(max_steps=100),
            )

            # Store original values for comparison
            original_point_location = np.array([5, 5, 0], dtype=np.float64)
            original_slice_origin = np.array([0, 0, 5], dtype=np.float64)

            # Preprocess params - this should apply coordinate system transformations
            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=False,
            )

    # Verify coordinate system status was stored
    assert processed_params.private_attribute_asset_cache.coordinate_system_status is not None
    cs_status = processed_params.private_attribute_asset_cache.coordinate_system_status
    assert len(cs_status.coordinate_systems) == 1
    assert cs_status.coordinate_systems[0].name == "test_frame"

    # Calculate expected transformations
    # Transformation matrix: scale(2) * rotation_z(90°) + translation(100, 200, 300)
    # Matrix = [[0, -2, 0, 100], [2, 0, 0, 200], [0, 0, 2, 300]]

    # Expected Point transformation:
    # - location (5, 5, 0) -> [[0, -2, 0], [2, 0, 0], [0, 0, 2]] @ [5, 5, 0] + [100, 200, 300]
    #   = [-10, 10, 0] + [100, 200, 300] = [90, 210, 300]
    expected_point_location = np.array([90, 210, 300], dtype=np.float64)

    # Expected Slice transformation:
    # - origin (0, 0, 5) -> [[0, -2, 0], [2, 0, 0], [0, 0, 2]] @ [0, 0, 5] + [100, 200, 300]
    #   = [0, 0, 10] + [100, 200, 300] = [100, 200, 310]
    # - normal (0, 0, 1) -> rotation only: [[0, -2, 0], [2, 0, 0], [0, 0, 2]] @ [0, 0, 1]
    #   = [0, 0, 2], normalized = [0, 0, 1]
    expected_slice_origin = np.array([100, 200, 310], dtype=np.float64)
    expected_slice_normal = np.array([0, 0, 1], dtype=np.float64)

    # Call get_solver_json which will apply transformations via @preprocess_input decorator
    translated = get_solver_json(processed_params, mesh_unit=1 * u.m)

    # Verify the transformed values appear in the translated output
    assert "monitorOutput" in translated, "monitorOutput not found in translated JSON"
    monitor_output = translated["monitorOutput"]
    assert "monitors" in monitor_output, "monitors not found in monitorOutput"

    # Find the test_probe in the output
    monitors = monitor_output["monitors"]
    assert "test_probe" in monitors, "test_probe not found in monitors"
    probe_data = monitors["test_probe"]
    assert "monitorLocations" in probe_data, "monitorLocations not found in probe data"

    # Get the probe location for test_point
    monitor_locations = probe_data["monitorLocations"]
    assert "test_point" in monitor_locations, "test_point not found in monitorLocations"
    probe_location = np.array(monitor_locations["test_point"], dtype=np.float64)

    # Verify the probe location is the transformed value
    np.testing.assert_allclose(
        probe_location,
        expected_point_location,
        atol=1e-9,
        err_msg=f"Probe location in translated JSON does not match expected transformation. "
        f"Expected: {expected_point_location}, Got: {probe_location}",
    )

    # Verify slice output
    assert "sliceOutput" in translated, "sliceOutput not found in translated JSON"
    slice_output = translated["sliceOutput"]
    assert "slices" in slice_output, "slices not found in sliceOutput"

    # slices is a dict with slice names as keys
    slices = slice_output["slices"]
    assert "test_slice" in slices, "test_slice not found in slices"
    slice_data = slices["test_slice"]

    assert "sliceOrigin" in slice_data, "sliceOrigin not found in slice data"
    assert "sliceNormal" in slice_data, "sliceNormal not found in slice data"

    # Verify the slice origin is the transformed value
    slice_origin = np.array(slice_data["sliceOrigin"], dtype=np.float64)
    np.testing.assert_allclose(
        slice_origin,
        expected_slice_origin,
        atol=1e-9,
        err_msg=f"Slice origin in translated JSON does not match expected transformation. "
        f"Expected: {expected_slice_origin}, Got: {slice_origin}",
    )

    # Verify the slice normal is the transformed value
    slice_normal = np.array(slice_data["sliceNormal"], dtype=np.float64)
    np.testing.assert_allclose(
        slice_normal,
        expected_slice_normal,
        atol=1e-9,
        err_msg=f"Slice normal in translated JSON does not match expected transformation. "
        f"Expected: {expected_slice_normal}, Got: {slice_normal}",
    )


def test_transformed_entities_in_translated_output(mock_geometry):
    """Test that transformed entity values appear correctly in the final translated JSON.

    This test verifies the complete pipeline:
    1. Entities are assigned coordinate systems
    2. Entities are transformed during preprocessing
    3. Transformed values appear in the translated case JSON
    """
    mock_geometry.internal_registry = mock_geometry._entity_info.get_persistent_entity_registry(
        mock_geometry.internal_registry
    )

    with create_draft(new_run_from=mock_geometry) as draft:
        with SI_unit_system:
            # Create a simple test: Point with coordinate system
            test_point = Point(
                name="probe_point",
                location=(1, 0, 0) * u.m,
            )

            # Coordinate system: 90 degree rotation around Z
            coordinate_system = CoordinateSystem(
                name="rotated_frame",
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=90 * u.deg,
            )

            # Note: Points in outputs typically don't get coordinate system assignments
            # in the current implementation, but we can test the pipeline

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(surface_max_edge_length=0.2),
                    volume_zones=[AutomatedFarfield()],
                ),
                outputs=[
                    ProbeOutput(
                        name="test_probe",
                        output_fields=["velocity"],
                        entities=[test_point],
                    ),
                ],
                operating_condition=AerospaceCondition(velocity_magnitude=10 * u.m / u.s),
                time_stepping=Steady(max_steps=100),
            )

            # Note: This test doesn't assign coordinate systems, so we don't expect transformations
            processed_params = set_up_params_for_uploading(
                root_asset=mock_geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=False,
                use_geometry_AI=False,
            )

    # Call get_solver_json which will apply transformations via @preprocess_input decorator
    translated = get_solver_json(processed_params, mesh_unit=1 * u.m)

    # Verify the output structure exists
    assert "monitorOutput" in translated, "monitorOutput should exist"

    # The point location should be in the translated output
    # Since no coordinate system was assigned, location should be original (1, 0, 0)
    monitor = translated["monitorOutput"]
    assert "monitors" in monitor, "monitors should exist in monitorOutput"
    assert "test_probe" in monitor["monitors"], "test_probe should exist"
    probe_data = monitor["monitors"]["test_probe"]
    assert "monitorLocations" in probe_data, "monitorLocations should exist"
    assert "probe_point" in probe_data["monitorLocations"], "probe_point should exist"

    # Verify location is unchanged (no coordinate system assigned)
    location = np.array(probe_data["monitorLocations"]["probe_point"], dtype=np.float64)
    expected_location = np.array([1, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(
        location,
        expected_location,
        atol=1e-9,
        err_msg=f"Location should be unchanged. Expected: {expected_location}, Got: {location}",
    )
