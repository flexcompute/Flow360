"""The shipped BET disk example files must keep loading through ``from_file``.

``examples/post_processing/special_features/bet_disk.py`` and
``examples/advanced_simulations/rotorcraft/BET_eVTOL.py`` drop
``BETDisk.from_file(...)`` straight into ``models=[...]`` with no ``entities``
argument, relying on the cylinders written inside each file. Neither example runs
in CI (they need a cloud project), so nothing else covers that expectation.
"""

import os

import pytest
from flow360_schema.models.entities.volume_entities import Cylinder
from flow360_schema.models.simulation.models.volume_models import BETDisk

import flow360 as fl

EXAMPLES_DIR = os.path.join(os.path.dirname(fl.__file__), "examples")

# (file, expected cylinder count) — versioned eVTOL dumps and the unversioned
# disk0, which predates every updater milestone.
EXAMPLE_DISKS = [
    ("betExampleData/disk0.json", 1),
    ("betEVTOL/disk13.json", 2),
    ("betEVTOL/disk24.json", 2),
    ("betEVTOL/disk57.json", 2),
    ("betEVTOL/disk68.json", 2),
]


@pytest.mark.parametrize("relative_path, cylinder_count", EXAMPLE_DISKS)
def test_shipped_bet_disk_example_keeps_its_cylinders(relative_path, cylinder_count):
    disk = BETDisk.from_file(os.path.join(EXAMPLES_DIR, relative_path))

    cylinders = disk.entities.stored_entities
    assert len(cylinders) == cylinder_count
    assert all(isinstance(cylinder, Cylinder) for cylinder in cylinders)
    # The disk itself has to arrive complete, not just its entities.
    assert disk.number_of_blades > 0
    assert disk.twists and disk.chords and disk.sectional_polars


def test_bet_disk_from_file_cylinders_reach_the_entity_pool():
    """Loading the cylinders is only half of what the examples need.

    Definitions live in the project entity pool and the params wire format carries
    only references to them, so a cylinder that arrives from a file has to be
    collected as a draft entity when the disk is put into params — otherwise the
    example would load fine and then fail on submit.
    """
    disk = BETDisk.from_file(os.path.join(EXAMPLES_DIR, "betEVTOL/disk13.json"))

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            operating_condition=fl.AerospaceCondition(velocity_magnitude=70),
            models=[fl.Fluid(), disk],
        )

    collected = list(params.used_entity_registry.view(Cylinder))
    assert sorted(cylinder.name for cylinder in collected) == ["BET 1", "BET 3"]
    assert {cylinder.private_attribute_id for cylinder in collected} == {
        cylinder.private_attribute_id for cylinder in disk.entities.stored_entities
    }
