"""Test the integration of python client with various metadatas."""

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import AngularVelocity, Rotation
from flow360.component.simulation.primitives import Cylinder, GhostSurface, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def get_volume_mesh_metadata():
    """plateASI case"""
    return {
        "zones": {
            "farFieldBlock": {
                "boundaryNames": [
                    "farFieldBlock/farField",
                    "farFieldBlock/rotIntf",
                    "farFieldBlock/slipWall",
                ],
                "donorInterfaceNames": ["plateBlock/rotIntf"],
                "donorZoneNames": ["plateBlock"],
                "receiverInterfaceNames": ["farFieldBlock/rotIntf"],
            },
            "plateBlock": {
                "boundaryNames": [
                    "plateBlock/noSlipWall",
                    "plateBlock/rotIntf",
                    # "plateBlock/slipWall", # We do not support split boundary across blocks
                ],
                "donorInterfaceNames": ["farFieldBlock/rotIntf"],
                "donorZoneNames": ["farFieldBlock"],
                "receiverInterfaceNames": ["plateBlock/rotIntf"],
            },
        }
    }


def test_update_zone_info_from_volume_mesh(get_volume_mesh_metadata):
    # Param is generated before the volume mesh metadata is available AKA the param generated the volume mesh.
    # (Though the volume meshing params are skipped here)
    with SI_unit_system:
        params = SimulationParams(
            models=[
                Rotation(
                    volumes=[
                        Cylinder(
                            name="plateBlock",
                            axis=(0, 2, 0),
                            center=(0, 1, 2),
                            height=0.2,
                            outer_radius=5,
                        )
                    ],
                    spec=AngularVelocity(200 * u.deg / u.hour),
                ),
                SlipWall(entities=[Surface(name="slipWall")]),
                Wall(entities=[Surface(name="noSlipWall")]),
                Freestream(entities=[GhostSurface(name="farField")]),
            ]
        )
    params._update_zone_info_from_volume_mesh(get_volume_mesh_metadata)

    assert isinstance(
        params.private_attribute_asset_cache.registry.find_by_name("plateBlock"),
        Cylinder,
    )
    assert (
        params.private_attribute_asset_cache.registry.find_by_name(
            "plateBlock"
        ).private_attribute_zone_boundary_names.items
        == get_volume_mesh_metadata["zones"]["plateBlock"]["boundaryNames"]
    )
    assert (
        params.private_attribute_asset_cache.registry.find_by_name(
            "slipWall"
        ).private_attribute_full_name
        == "farFieldBlock/slipWall"
    )
    assert (
        params.private_attribute_asset_cache.registry.find_by_name(
            "noSlipWall"
        ).private_attribute_full_name
        == "plateBlock/noSlipWall"
    )
    assert (
        params.private_attribute_asset_cache.registry.find_by_name(
            "farField"
        ).private_attribute_full_name
        == "farFieldBlock/farField"
    )
