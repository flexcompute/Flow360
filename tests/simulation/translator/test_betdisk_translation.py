import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.services import clear_context
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import imperial_unit_system
from tests.simulation.translator.utils.xv15_bet_disk_helper import (
    createBETDiskSteady,
    createBETDiskUnsteady,
    createSteadyTimeStepping,
)
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    create_param_base,
    createUnsteadyTimeStepping,
)


@pytest.fixture(autouse=True)
def reset_context():
    """Clear user variables from the context."""
    clear_context()


def _create_test_cylinder():
    return Cylinder(
        name="bet_zone",
        center=(0, 0, 0) * u.inch,
        axis=[0, 0, 1],
        outer_radius=150 * u.inch,
        height=15 * u.inch,
    )


def test_betdisk_unsteady_excludes_internal_fields():
    rpm = 588.50450
    params = create_param_base()
    bet_disk = createBETDiskUnsteady(
        cylinder_entity=_create_test_cylinder(), pitch_in_degree=10, rpm=rpm
    )
    params.models.append(bet_disk)
    params.time_stepping = createUnsteadyTimeStepping(rpm)

    translated = get_solver_json(params, mesh_unit=1 * u.inch)
    assert "BETDisks" in translated and len(translated["BETDisks"]) > 0
    bet_item = translated["BETDisks"][0]

    assert "initialBladeDirection" in bet_item
    assert "bladeLineChord" in bet_item


def test_betdisk_steady_excludes_internal_fields():
    rpm = 588.50450
    params = create_param_base()
    bet_disk = createBETDiskSteady(
        cylinder_entity=_create_test_cylinder(), pitch_in_degree=10, rpm=rpm
    )
    bet_disk = bet_disk.model_copy(
        update={
            "blade_line_chord": 25 * u.inch,
            "initial_blade_direction": (1, 0, 0),
        }
    )
    params.models.append(bet_disk)
    params.time_stepping = createSteadyTimeStepping()

    translated = get_solver_json(params, mesh_unit=1 * u.inch)
    assert "BETDisks" in translated and len(translated["BETDisks"]) > 0
    bet_item = translated["BETDisks"][0]

    assert "initialBladeDirection" not in bet_item
    assert "bladeLineChord" not in bet_item
