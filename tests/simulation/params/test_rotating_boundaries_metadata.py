"""Unit tests for rotating boundaries metadata update functionality."""

from flow360 import u
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationVolume,
)
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
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

        # Update entity full names first (simulating what happens before this function)
        from flow360.component.simulation.framework.param_utils import (
            _update_entity_full_name,
            _update_rotating_boundaries_with_metadata,
        )
        from flow360.component.simulation.primitives import (
            _SurfaceEntityBase,
            _VolumeEntityBase,
        )

        _update_entity_full_name(params, _VolumeEntityBase, volume_mesh_meta_data)
        _update_entity_full_name(params, _SurfaceEntityBase, volume_mesh_meta_data)

        # Call the function to update rotating boundaries
        _update_rotating_boundaries_with_metadata(params, volume_mesh_meta_data)

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

        # Update entity full names first
        from flow360.component.simulation.framework.param_utils import (
            _update_entity_full_name,
            _update_rotating_boundaries_with_metadata,
        )
        from flow360.component.simulation.primitives import (
            _SurfaceEntityBase,
            _VolumeEntityBase,
        )

        _update_entity_full_name(params, _VolumeEntityBase, volume_mesh_meta_data)
        _update_entity_full_name(params, _SurfaceEntityBase, volume_mesh_meta_data)

        # Call the function to update rotating boundaries
        _update_rotating_boundaries_with_metadata(params, volume_mesh_meta_data)

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
