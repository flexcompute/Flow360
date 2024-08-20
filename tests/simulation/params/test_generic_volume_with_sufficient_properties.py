import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import (
    AngleExpression,
    PorousMedium,
    Rotation,
)
from flow360.component.simulation.primitives import GenericVolume


def test_ensure_entities_have_sufficient_attributes():
    with pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axes' must specify `axes` to be used under `PorousMedium`.",
    ):

        PorousMedium(
            volumes=[GenericVolume(name="zone_with_no_axes")],
            darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
            forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
            volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
        )

    with pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `axis` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis")],
            spec=AngleExpression("0.45 * u.rad / u.s"),
        )

    with pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `center` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis", axis=[1, 2, 3])],
            spec=AngleExpression("0.45 * u.rad / u.s"),
        )
