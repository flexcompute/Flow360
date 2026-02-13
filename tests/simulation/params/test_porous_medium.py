import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import PorousMedium
from flow360.component.simulation.primitives import GenericVolume
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    ValidationContext,
)


def test_ensure_entities_have_sufficient_attributes():
    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )

    with (
        mock_context,
        pytest.raises(
            ValueError,
            match="Entity 'zone_with_no_axes' must specify `axes` to be used under `PorousMedium`.",
        ),
    ):

        PorousMedium(
            volumes=[GenericVolume(name="zone_with_no_axes")],
            darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
            forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
            volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
        )
