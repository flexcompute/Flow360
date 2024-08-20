import pytest

from flow360.component.simulation.models.volume_models import AngleExpression, Rotation
from flow360.component.simulation.primitives import GenericVolume


def test_ensure_entities_have_sufficient_attributes():

    with pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `axis` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis")],
            spec=AngleExpression("0.45 * t"),
        )

    with pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `center` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis", axis=[1, 2, 3])],
            spec=AngleExpression("0.45 * t"),
        )
