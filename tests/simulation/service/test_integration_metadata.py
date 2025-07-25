"""Test the integration of python client with various metadatas."""

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import (
    Freestream,
    Periodic,
    Rotational,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import AngularVelocity, Rotation
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def get_volume_mesh_metadata():
    """A realistic volume mesh metadata."""
    return {
        "zones": {
            "rotatingBlock-rotating_zone": {
                "boundaryNames": [
                    "rotatingBlock-rotating_zone/blade1",
                    "rotatingBlock-rotating_zone/blade2",
                    "rotatingBlock-rotating_zone/blade3",
                    "rotatingBlock-rotating_zone/blade4",
                    "rotatingBlock-rotating_zone/blade5",
                    "rotatingBlock-rotating_zone/hub",
                    "rotatingBlock-rotating_zone/slidingInterface-rotating_zone",
                ],
                "donorInterfaceNames": ["stationaryBlock/slidingInterface-rotating_zone"],
                "donorZoneNames": ["stationaryBlock"],
                "receiverInterfaceNames": [
                    "rotatingBlock-rotating_zone/slidingInterface-rotating_zone"
                ],
            },
            "stationaryBlock": {
                "boundaryNames": [
                    "stationaryBlock/farfield",
                    "stationaryBlock/slidingInterface-rotating_zone",
                ],
                "donorInterfaceNames": [
                    "rotatingBlock-rotating_zone/slidingInterface-rotating_zone"
                ],
                "donorZoneNames": ["rotatingBlock-rotating_zone"],
                "receiverInterfaceNames": ["stationaryBlock/slidingInterface-rotating_zone"],
            },
        }
    }


def test_update_zone_info_from_volume_mesh(get_volume_mesh_metadata):
    # Param is generated before the volume mesh metadata is available AKA the param generated the volume mesh.
    # (Though the volume meshing params are skipped here)
    with SI_unit_system:
        auto_farfield = AutomatedFarfield(name="my_farfield")
        params = SimulationParams(
            operating_condition=AerospaceCondition.from_mach(mach=0.2),
            meshing=MeshingParams(
                volume_zones=[auto_farfield],
            ),
            models=[
                Rotation(
                    volumes=[
                        Cylinder(
                            name="rotating_zone",
                            axis=(0, 2, 0),
                            center=(0, 1, 2),
                            height=0.2,
                            outer_radius=5,
                        )
                    ],
                    spec=AngularVelocity(200 * u.deg / u.hour),
                ),
                Wall(
                    entities=[
                        Surface(name="blade1"),
                        Surface(name="blade2"),
                        Surface(name="blade3"),
                    ]
                ),
                Periodic(
                    surface_pairs=[
                        (
                            Surface(name="blade4"),
                            Surface(name="blade5"),
                        )
                    ],
                    spec=Rotational(),
                ),
                Freestream(entities=[auto_farfield.farfield]),
            ],
        )
    params._update_param_with_actual_volume_mesh_meta(get_volume_mesh_metadata)
    my_reg = params.used_entity_registry
    assert isinstance(
        my_reg.find_by_naming_pattern(pattern="rotating_zone", enforce_output_as_list=False),
        Cylinder,
    )
    assert (
        my_reg.find_by_naming_pattern(
            pattern="rotating_zone", enforce_output_as_list=False
        ).private_attribute_zone_boundary_names.items
        == get_volume_mesh_metadata["zones"]["rotatingBlock-rotating_zone"]["boundaryNames"]
    )
    assert (
        my_reg.find_by_naming_pattern(
            pattern="blade1", enforce_output_as_list=False
        ).private_attribute_full_name
        == "rotatingBlock-rotating_zone/blade1"
    )
    assert (
        my_reg.find_by_naming_pattern(
            pattern="blade3", enforce_output_as_list=False
        ).private_attribute_full_name
        == "rotatingBlock-rotating_zone/blade3"
    )
    assert (
        my_reg.find_by_naming_pattern(
            pattern="farfield", enforce_output_as_list=False
        ).private_attribute_full_name
        == "stationaryBlock/farfield"
    )
    assert (
        my_reg.find_by_naming_pattern(
            pattern="__farfield_zone_name_not_properly_set_yet", enforce_output_as_list=False
        ).private_attribute_full_name
        == "stationaryBlock"
    )

    translated = get_solver_json(params, mesh_unit="m")

    assert list(translated["volumeZones"].keys()) == ["rotatingBlock-rotating_zone"]
    assert "rotatingBlock-rotating_zone/blade4" in translated["boundaries"]
    assert translated["boundaries"]["rotatingBlock-rotating_zone/blade4"] == {
        "type": "RotationallyPeriodic",
        "pairedPatchName": "rotatingBlock-rotating_zone/blade5",
    }
