import flow360.component.simulation.units as u
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemStatus,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.volume_models import (
    BETDisk,
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
)
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    apply_coordinate_system_transformations,
)
from flow360.component.simulation.unit_system import SI_unit_system


def test_apply_coordinate_system_transformations_skips_frozen_non_entity_sequences():
    with SI_unit_system:
        cylinder = Cylinder(
            name="bet_disk",
            center=(0, 0, 0) * u.m,
            axis=(0, 0, 1),
            height=1 * u.m,
            outer_radius=1 * u.m,
        )
        bet_disk = BETDisk(
            entities=cylinder,
            rotation_direction_rule="leftHand",
            number_of_blades=3,
            omega=100 * u.rpm,
            chord_ref=1 * u.m,
            n_loading_nodes=20,
            mach_numbers=[0],
            reynolds_numbers=[1000000],
            twists=[BETDiskTwist(radius=0 * u.m, twist=0 * u.deg)],
            chords=[BETDiskChord(radius=0 * u.m, chord=1 * u.m)],
            alphas=[-2, 0, 2] * u.deg,
            sectional_radiuses=[0.25, 0.5] * u.m,
            sectional_polars=[
                BETDiskSectionalPolar(
                    lift_coeffs=[[[0.1, 0.2, 0.3]]],
                    drag_coeffs=[[[0.01, 0.02, 0.03]]],
                ),
                BETDiskSectionalPolar(
                    lift_coeffs=[[[0.15, 0.25, 0.35]]],
                    drag_coeffs=[[[0.015, 0.025, 0.035]]],
                ),
            ],
        )
        coordinate_system = CoordinateSystem(name="cs", translation=(1, 2, 3) * u.m)
        coordinate_system_status = CoordinateSystemStatus(
            coordinate_systems=[coordinate_system],
            parents=[],
            assignments=[
                CoordinateSystemAssignmentGroup(
                    coordinate_system_id=coordinate_system.private_attribute_id,
                    entities=[
                        CoordinateSystemEntityRef(
                            entity_type="Cylinder",
                            entity_id=cylinder.private_attribute_id,
                        )
                    ],
                )
            ],
        )
        params = SimulationParams(
            models=[bet_disk],
            private_attribute_asset_cache=AssetCache(
                coordinate_system_status=coordinate_system_status,
            ),
        )

    apply_coordinate_system_transformations(params)

    transformed_bet_disk = params.models[0]
    transformed_cylinder = transformed_bet_disk.entities.stored_entities[0]

    assert all(transformed_cylinder.center == [1, 2, 3] * u.m)
    assert transformed_bet_disk.mach_numbers == [0]
    assert transformed_bet_disk.twists[0].radius == 0 * u.m
    assert transformed_bet_disk.chords[0].chord == 1 * u.m
