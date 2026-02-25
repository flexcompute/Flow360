import re

import pytest

from flow360.component.simulation.models.volume_models import ActuatorDisk, ForcePerArea
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import (
    actuator_disk_translator,
)
from flow360.component.simulation.unit_system import SI_unit_system, u


def test_actuator_disk():
    with SI_unit_system:
        fpa = ForcePerArea(radius=[0, 1, 2, 4], thrust=[1, 1, 2, 2], circumferential=[1, 1, 3, 4])
        assert fpa

        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(5, 0, 0),
            center=(1.2, 2.3, 3.4),
            height=3.0,
            outer_radius=5.0,
        )

        ad = ActuatorDisk(volumes=[my_cylinder_1], force_per_area=fpa)
        assert ad

        with pytest.raises(
            ValueError,
            match=re.escape(
                "length of radius, thrust, circumferential must be the same, but got: "
                + "len(radius)=3, len(thrust)=2, len(circumferential)=2"
            ),
        ):
            fpa = ForcePerArea(radius=[0, 1, 3], thrust=[1, 1], circumferential=[1, 1])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "length of radius, thrust, circumferential must be the same, but got: "
            + "len(radius)=3, len(thrust)=2, len(circumferential)=2"
        ),
    ):
        fpa = ForcePerArea(
            radius=[0, 1, 3] * u.m, thrust=[1, 1] * u.Pa, circumferential=[1, 1] * u.Pa
        )


def test_actuator_disk_from_json():
    data = {
        "entities": {
            "stored_entities": [
                {
                    "private_attribute_entity_type_name": "Cylinder",
                    "name": "my_cylinder-1",
                    "axis": (1.0, 0.0, 0.0),
                    "center": {"value": (1.2, 2.3, 3.4), "units": "m"},
                    "height": {"value": 3.0, "units": "m"},
                    "outer_radius": {"value": 5.0, "units": "m"},
                }
            ]
        },
        "force_per_area": {
            "radius": {"value": (0.0, 1.0, 2.0, 4.0), "units": "m"},
            "thrust": {"value": (1.0, 1.0, 2.0, 2.0), "units": "N/m**2"},
            "circumferential": {"value": (1.0, 1.0, 3.0, 4.0), "units": "N/m**2"},
        },
        "type": "ActuatorDisk",
    }
    ad = ActuatorDisk(**data)

    assert ad.force_per_area.radius[2].value == 2
    assert ad.force_per_area.radius[2].units == u.m
    assert ad.entities.stored_entities[0].center[0].value == 1.2


def test_actuator_disk_duplicate_cylinder_names(mock_validation_context):
    with SI_unit_system:
        fpa = ForcePerArea(radius=[0, 1, 2, 4], thrust=[1, 1, 2, 2], circumferential=[1, 1, 3, 4])
        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(5, 0, 0),
            center=(1.2, 2.3, 3.4),
            height=3.0,
            outer_radius=5.0,
        )

        my_cylinder_2 = Cylinder(
            name="my_cylinder-2",
            axis=(5, 0, 0),
            center=(2.2, 2.3, 3.4),
            height=3.0,
            outer_radius=5.0,
        )

        ad = ActuatorDisk(volumes=[my_cylinder_1, my_cylinder_2], force_per_area=fpa)
        sm = SimulationParams(models=[ad])

        assert sm

        with (
            mock_validation_context,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"The ActuatorDisk cylinder name `my_cylinder-1` at index 1 in model `Actuator disk` "
                    "has already been used. Please use unique Cylinder entity names among all "
                    "ActuatorDisk instances."
                ),
            ),
        ):
            ad_duplicate = ActuatorDisk(volumes=[my_cylinder_1, my_cylinder_1], force_per_area=fpa)
            sm = SimulationParams(models=[ad_duplicate])


def _make_actuator_disk(reference_velocity=None):
    """Helper to create an ActuatorDisk with optional reference_velocity."""
    kwargs = dict(
        volumes=[
            Cylinder(
                name="test_disk",
                axis=(1, 0, 0),
                center=(0, 0, 0),
                height=3.0,
                outer_radius=5.0,
            ),
        ],
        force_per_area=ForcePerArea(radius=[0, 1, 2], thrust=[10, 8, 0], circumferential=[2, 3, 0]),
    )
    if reference_velocity is not None:
        kwargs["reference_velocity"] = reference_velocity
    return ActuatorDisk(**kwargs)


def test_actuator_disk_without_reference_velocity():
    with SI_unit_system:
        ad = _make_actuator_disk()
        assert ad.reference_velocity is None


def test_actuator_disk_with_reference_velocity():
    with SI_unit_system:
        ad = _make_actuator_disk(reference_velocity=(10.0, 0.0, 0.0) * u.m / u.s)
        assert ad.reference_velocity is not None


def test_actuator_disk_reference_velocity_different_units():
    with SI_unit_system:
        ad = _make_actuator_disk(reference_velocity=(32.8084, 0.0, 0.0) * u.ft / u.s)
        assert ad.reference_velocity is not None


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
