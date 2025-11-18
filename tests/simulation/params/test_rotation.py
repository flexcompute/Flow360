import pytest

from flow360 import u
from flow360.component.simulation.models.surface_models import Wall, WallRotation
from flow360.component.simulation.models.volume_models import AngleExpression, Rotation
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import GenericVolume, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def test_ensure_entities_have_sufficient_attributes(mock_validation_context):

    with mock_validation_context, pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `axis` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis")],
            spec=AngleExpression("0.45 * t"),
        )

    with mock_validation_context, pytest.raises(
        ValueError,
        match="Entity 'zone_with_no_axis' must specify `center` to be used under `Rotation`.",
    ):

        Rotation(
            volumes=[GenericVolume(name="zone_with_no_axis", axis=[1, 2, 3])],
            spec=AngleExpression("0.45 * t"),
        )


def test_wall_angular_velocity():
    my_wall_surface = Surface(name="my_wall")
    Wall(
        surfaces=[my_wall_surface],
        velocity=WallRotation(axis=(0, 0, 1), center=(1, 2, 3) * u.m, angular_velocity=100 * u.rpm),
        use_wall_function=True,
    )


def test_rotation_expression_with_t_seconds():

    with pytest.raises(
        ValueError,
        match=r"Syntax error in expression `0.45 \* sin\(0.2\*\*\*tss`: invalid syntax\.",
    ):
        Rotation(
            volumes=[GenericVolume(name="zone_1", axis=[1, 2, 3], center=(1, 1, 1) * u.cm)],
            spec=AngleExpression("0.45 * sin(0.2***tss"),
        )

    with pytest.raises(
        ValueError,
        match=r"Unexpected variable `taa` found.",
    ):
        Rotation(
            volumes=[GenericVolume(name="zone_1", axis=[1, 2, 3], center=(1, 1, 1) * u.cm)],
            spec=AngleExpression("0.45 + taa"),
        )

    with pytest.raises(
        ValueError,
        match=r"t_seconds must be used as a multiplicative factor, not directly added/subtracted with a number.",
    ):
        Rotation(
            volumes=[GenericVolume(name="zone_1", axis=[1, 2, 3], center=(1, 1, 1) * u.cm)],
            spec=AngleExpression("sin(0.45 + t_seconds)"),
        )

    Rotation(
        volumes=[GenericVolume(name="zone_1", axis=[1, 2, 3], center=(1, 1, 1) * u.cm)],
        spec=AngleExpression(
            "-180/pi * atan(2 * 3.00 * 20.00 * 2.00/180*pi * "
            "cos(2.00/180*pi * sin(0.05877271 * t_seconds)) * cos(0.05877271 * t_seconds) / 50.00) +"
            " 2 * 2.00 * sin(0.05877271 * t_seconds) - 2.00 * sin(0.05877271 * t_seconds)"
        ),
    )

    with SI_unit_system:
        op = AerospaceCondition(velocity_magnitude=10)
        params = SimulationParams(
            operating_condition=op,
            models=[
                Rotation(
                    volumes=[GenericVolume(name="zone_1", axis=[1, 2, 3], center=(1, 1, 1) * u.cm)],
                    spec=AngleExpression("0.45 * sin(0.2*t_seconds)"),
                ),
                Rotation(
                    volumes=[
                        GenericVolume(name="zone_2", axis=[11, 2, 3], center=(1, 1, 1) * u.cm)
                    ],
                    spec=AngleExpression("0.45 * sin(0.5*t_seconds  +0.2)"),
                ),
            ],
        )

    flow360_time_in_seconds = 0.025
    processed_params = params._preprocess(
        mesh_unit=(op.thermal_state.speed_of_sound.value * flow360_time_in_seconds) * u.m
    )
    assert (
        processed_params.models[0].spec.value == f"0.45 * sin(0.2*({flow360_time_in_seconds} * t))"
    )
    assert (
        processed_params.models[1].spec.value
        == f"0.45 * sin(0.5*({flow360_time_in_seconds} * t)  +0.2)"
    )
