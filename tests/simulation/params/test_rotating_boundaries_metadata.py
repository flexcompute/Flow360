"""Unit tests for rotating boundaries metadata update functionality."""

from flow360 import u
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationVolume,
)
from flow360.component.simulation.models.surface_models import Wall, WallRotation
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def test_update_rotating_boundaries_with_metadata():
    """Test updating rotating boundaries with metadata from volume mesh."""
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/sphere.lb8.ugrid",
                ],
                "donorInterfaceNames": ["intersectingCylinder/inverted-intersectingCylinder"],
                "donorZoneNames": ["intersectingCylinder"],
                "receiverInterfaceNames": ["farfield/slidingInterface-intersectingCylinder"],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder",
                ],
                "donorInterfaceNames": ["farfield/slidingInterface-intersectingCylinder"],
                "donorZoneNames": ["farfield"],
                "receiverInterfaceNames": ["intersectingCylinder/inverted-intersectingCylinder"],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")

        # Create RotationVolume with enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create Wall model for the sphere surface
        wall_model = Wall(
            entities=[sphere_surface],
        )

        # Create simulation params
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            models=[wall_model],
        )

        # Update using the unified volume mesh metadata API
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # Verify that the enclosed_entity was updated to point to __rotating patch
        updated_entity = rotation_volume.enclosed_entities.stored_entities[0]
        assert (
            updated_entity.full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )
        assert updated_entity.name == "sphere.lb8.ugrid__rotating_intersectingCylinder"

        # Verify that a new Wall model was created for the __rotating patch
        # Original model should still exist
        assert len(params.models) > 1
        wall_models = [m for m in params.models if isinstance(m, Wall)]
        assert len(wall_models) > 1

        # Find the model for the rotating boundary
        rotating_wall = None
        for model in wall_models:
            if model.entities and model.entities.stored_entities:
                entity_full_name = model.entities.stored_entities[0].full_name
                if "__rotating" in entity_full_name:
                    rotating_wall = model
                    break

        assert rotating_wall is not None
        assert (
            rotating_wall.entities.stored_entities[0].full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )


def test_update_rotating_boundaries_with_stationary_entities():
    """Test updating rotating boundaries with stationary_enclosed_entities (velocity=0)."""
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/sphere.lb8.ugrid",
                ],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder",
                ],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")

        # Create RotationVolume with enclosed_entities and stationary_enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface],
            stationary_enclosed_entities=[sphere_surface],  # Mark as stationary
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create Wall model for the sphere surface with non-zero velocity
        wall_model = Wall(
            entities=[sphere_surface],
            velocity=[1, 0, 0],  # Non-zero velocity
        )

        # Create simulation params
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            models=[wall_model],
        )

        # Update using the unified volume mesh metadata API
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # Verify that the stationary_enclosed_entity was updated
        updated_stationary_entity = rotation_volume.stationary_enclosed_entities.stored_entities[0]
        assert (
            updated_stationary_entity.full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )

        # Verify that a new Wall model was created for the __rotating patch with velocity=0
        wall_models = [m for m in params.models if isinstance(m, Wall)]
        rotating_wall = None
        for model in wall_models:
            if model.entities and model.entities.stored_entities:
                entity_full_name = model.entities.stored_entities[0].full_name
                if "__rotating" in entity_full_name:
                    rotating_wall = model
                    break

        assert rotating_wall is not None
        # Verify velocity is set to zero for stationary entities (as string expressions)
        assert rotating_wall.velocity == ("0", "0", "0")


def test_multiple_entities_partial_rotating_patches():
    """Test multiple entities in enclosed_entities where only some have __rotating patches."""
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/sphere.lb8.ugrid",
                    "farfield/other_surface",
                ],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder",
                    # Note: other_surface does NOT have a __rotating patch
                ],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")
        other_surface = Surface(name="other_surface")

        # Create RotationVolume with multiple enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface, other_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create Wall models for both surfaces
        wall_model_sphere = Wall(entities=[sphere_surface], velocity=[1, 0, 0])
        wall_model_other = Wall(entities=[other_surface], velocity=[0, 1, 0])

        # Create simulation params
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            models=[wall_model_sphere, wall_model_other],
        )

        # Update using the unified volume mesh metadata API
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # Verify that only sphere_surface was updated (has __rotating patch)
        updated_entities = rotation_volume.enclosed_entities.stored_entities
        sphere_updated = None
        other_updated = None

        for entity in updated_entities:
            if entity.name == "sphere.lb8.ugrid__rotating_intersectingCylinder":
                sphere_updated = entity
            elif entity.name == "other_surface":
                other_updated = entity

        assert sphere_updated is not None
        assert (
            sphere_updated.full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )

        # other_surface should remain unchanged (no __rotating patch found)
        assert other_updated is not None
        assert other_updated.full_name == "farfield/other_surface"
        assert other_updated.name == "other_surface"

        # Verify that only one new Wall model was created (for sphere, not other_surface)
        wall_models = [m for m in params.models if isinstance(m, Wall)]
        rotating_walls = [
            m
            for m in wall_models
            if m.entities
            and m.entities.stored_entities
            and "__rotating" in m.entities.stored_entities[0].full_name
        ]
        assert len(rotating_walls) == 1
        assert (
            rotating_walls[0].entities.stored_entities[0].full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )


def test_no_wall_model_for_entity():
    """Test that entities without Wall models are handled correctly."""
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/sphere.lb8.ugrid",
                ],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder",
                ],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")

        # Create RotationVolume with enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create simulation params WITHOUT any Wall models
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            models=[],  # No Wall models
        )

        # Update using the unified volume mesh metadata API
        # This should not raise an error even though there's no Wall model
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # Verify that the enclosed_entity was still updated to point to __rotating patch
        updated_entity = rotation_volume.enclosed_entities.stored_entities[0]
        assert (
            updated_entity.full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )

        # Verify that no new Wall models were created (since there were none to begin with)
        wall_models = [m for m in params.models if isinstance(m, Wall)]
        assert len(wall_models) == 0


def test_wall_model_with_wall_rotation():
    """Test Wall model with WallRotation velocity instead of a simple velocity vector."""
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/sphere.lb8.ugrid",
                ],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder",
                ],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")

        # Create RotationVolume with enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create Wall model with WallRotation velocity
        wall_rotation = WallRotation(
            axis=(0, 0, 1),
            center=(0, 0, 0) * u.m,
            angular_velocity=100 * u.rpm,
        )
        wall_model = Wall(
            entities=[sphere_surface],
            velocity=wall_rotation,
        )

        # Create simulation params
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            models=[wall_model],
        )

        # Update using the unified volume mesh metadata API
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # Verify that a new Wall model was created for the __rotating patch
        wall_models = [m for m in params.models if isinstance(m, Wall)]
        rotating_wall = None
        for model in wall_models:
            if model.entities and model.entities.stored_entities:
                entity_full_name = model.entities.stored_entities[0].full_name
                if "__rotating" in entity_full_name:
                    rotating_wall = model
                    break

        assert rotating_wall is not None
        assert (
            rotating_wall.entities.stored_entities[0].full_name
            == "intersectingCylinder/sphere.lb8.ugrid__rotating_intersectingCylinder"
        )

        # Verify that WallRotation velocity is preserved (not converted to tuple)
        assert isinstance(rotating_wall.velocity, WallRotation)
        assert rotating_wall.velocity.axis == (0, 0, 1)
        assert all(rotating_wall.velocity.center == (0, 0, 0) * u.m)
        assert rotating_wall.velocity.angular_velocity == 100 * u.rpm


def test_surface_output_expanded_while_rotation_volume_filtered():
    """
    Test that SurfaceOutput entities are expanded to include all split versions,
    while RotationVolume.enclosed_entities are filtered to only keep __rotating patches.

    This verifies the key behavior difference:
    - SurfaceOutput: expands to include BOTH farfield/surface AND zone/__rotating versions
    - RotationVolume.enclosed_entities: filtered to ONLY keep __rotating version
    """
    volume_mesh_meta_data = {
        "zones": {
            "farfield": {
                "boundaryNames": [
                    "farfield/farfield",
                    "farfield/slidingInterface-intersectingCylinder",
                    "farfield/blade",  # Original surface in farfield zone
                ],
            },
            "intersectingCylinder": {
                "boundaryNames": [
                    "intersectingCylinder/inverted-intersectingCylinder",
                    "intersectingCylinder/blade__rotating_intersectingCylinder",  # __rotating patch
                ],
            },
        }
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        blade_surface = Surface(name="blade")

        # Create RotationVolume with enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[blade_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ]
        )

        # Create operating condition
        op = AerospaceCondition(velocity_magnitude=10)

        # Create SurfaceOutput for the same blade surface
        surface_output = SurfaceOutput(
            entities=[blade_surface],
            output_fields=["Cp", "Cf"],
        )

        # Create simulation params
        params = SimulationParams(
            meshing=meshing,
            operating_condition=op,
            outputs=[surface_output],
        )

        # Update using the unified volume mesh metadata API
        params._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)

        # === Verify RotationVolume.enclosed_entities is FILTERED ===
        # Should only have the __rotating version
        enclosed_entities = rotation_volume.enclosed_entities.stored_entities
        assert len(enclosed_entities) == 1
        assert (
            enclosed_entities[0].full_name
            == "intersectingCylinder/blade__rotating_intersectingCylinder"
        )
        assert enclosed_entities[0].name == "blade__rotating_intersectingCylinder"

        # === Verify SurfaceOutput.entities is EXPANDED ===
        # Should have BOTH the farfield version AND the __rotating version
        output_entities = surface_output.entities.stored_entities
        assert len(output_entities) == 2

        # Collect full_names for verification
        output_full_names = {e.full_name for e in output_entities}
        assert "farfield/blade" in output_full_names
        assert "intersectingCylinder/blade__rotating_intersectingCylinder" in output_full_names
