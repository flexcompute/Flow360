import unyt as u

from flow360.component.simulation.models.volume_models import ActuatorDisk, ForcePerArea
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.translator.solver_translator import (
    actuator_disk_translator,
)
from flow360.component.simulation.unit_system import SI_unit_system


def _make_actuator_disk(reference_velocity=None):
    kwargs = {
        "volumes": [
            Cylinder(
                name="test_disk",
                axis=(1, 0, 0),
                center=(0, 0, 0),
                height=3.0,
                outer_radius=5.0,
            ),
        ],
        "force_per_area": ForcePerArea(
            radius=[0, 1, 2],
            thrust=[10, 8, 0],
            circumferential=[2, 3, 0],
        ),
    }
    if reference_velocity is not None:
        kwargs["reference_velocity"] = reference_velocity
    return ActuatorDisk(**kwargs)


def test_actuator_disk_translator_omits_reference_velocity_when_not_set():
    with SI_unit_system:
        ad = _make_actuator_disk()
        result = actuator_disk_translator(ad)
        assert "referenceVelocity" not in result
        assert "forcePerArea" in result


def test_actuator_disk_translator_includes_reference_velocity_when_set():
    with SI_unit_system:
        ad = _make_actuator_disk(reference_velocity=(10.0, 5.0, 2.0) * u.m / u.s)
        result = actuator_disk_translator(ad)
        assert "referenceVelocity" in result
        rv = result["referenceVelocity"]
        assert isinstance(rv, list)
        assert len(rv) == 3
