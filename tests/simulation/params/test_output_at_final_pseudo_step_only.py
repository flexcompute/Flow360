"""Tests for the output_at_final_pseudo_step_only toggle on monitor output classes."""

import re

import pydantic
import pytest
from flow360_schema.models.variables import solution

import flow360.component.simulation.units as u
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import MovingStatistic, ProbeOutput
from flow360.component.simulation.run_control.stopping_criterion import (
    StoppingCriterion,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.validation.validation_context import TimeSteppingType


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


# ---------------------------------------------------------------------------
# StoppingCriterion + toggle: rejected in steady, allowed in unsteady
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_field():
    return UserVariable(name="density_field", value=solution.density)


@pytest.fixture()
def probe_with_toggle(scalar_field):
    return ProbeOutput(
        name="probe_toggle",
        probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
        output_fields=[scalar_field],
        output_at_final_pseudo_step_only=True,
    )


def test_stopping_criterion_rejects_toggle_in_steady(
    scalar_field, probe_with_toggle, mock_validation_context
):
    message = re.escape(
        "A monitor output with `output_at_final_pseudo_step_only=True` cannot be "
        "referenced by a StoppingCriterion in a steady simulation."
    )
    mock_validation_context.info.output_dict = {
        probe_with_toggle.private_attribute_id: probe_with_toggle
    }
    mock_validation_context.info.time_stepping = TimeSteppingType.STEADY
    with (
        SI_unit_system,
        mock_validation_context,
        pytest.raises(pydantic.ValidationError, match=message),
    ):
        StoppingCriterion(
            monitor_field=scalar_field,
            monitor_output=probe_with_toggle,
            tolerance=0.01 * u.kg / u.m**3,
        )


def test_stopping_criterion_allows_toggle_in_unsteady(
    scalar_field, probe_with_toggle, mock_validation_context
):
    mock_validation_context.info.output_dict = {
        probe_with_toggle.private_attribute_id: probe_with_toggle
    }
    mock_validation_context.info.time_stepping = TimeSteppingType.UNSTEADY
    with SI_unit_system, mock_validation_context:
        criterion = StoppingCriterion(
            monitor_field=scalar_field,
            monitor_output=probe_with_toggle,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.monitor_output == probe_with_toggle.private_attribute_id


def test_stopping_criterion_allows_no_toggle_in_steady(scalar_field, mock_validation_context):
    """StoppingCriterion with toggle=False (default) should be fine in steady."""
    probe_no_toggle = ProbeOutput(
        name="probe_no_toggle",
        probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
        output_fields=[scalar_field],
    )
    mock_validation_context.info.output_dict = {
        probe_no_toggle.private_attribute_id: probe_no_toggle
    }
    mock_validation_context.info.time_stepping = TimeSteppingType.STEADY
    with SI_unit_system, mock_validation_context:
        criterion = StoppingCriterion(
            monitor_field=scalar_field,
            monitor_output=probe_no_toggle,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.monitor_output == probe_no_toggle.private_attribute_id


# ---------------------------------------------------------------------------
# Steady + toggle + MovingStatistic: rejected
# ---------------------------------------------------------------------------


def test_toggle_with_moving_statistic_rejected_in_steady_probe(mock_validation_context):
    message = re.escape(
        "`output_at_final_pseudo_step_only=True` with `moving_statistic` is not allowed "
        "for steady simulations (only one data point would be produced)."
    )
    mock_validation_context.info.time_stepping = TimeSteppingType.STEADY
    with pytest.raises(pydantic.ValidationError, match=message):
        with SI_unit_system, mock_validation_context:
            ProbeOutput(
                name="probe",
                probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
                output_fields=["Cp"],
                output_at_final_pseudo_step_only=True,
                moving_statistic=MovingStatistic(
                    method="mean", moving_window_size=10, start_step=100
                ),
            )


# ---------------------------------------------------------------------------
# Unsteady + toggle + MovingStatistic: allowed
# ---------------------------------------------------------------------------


def test_toggle_with_moving_statistic_allowed_in_unsteady(mock_validation_context):
    mock_validation_context.info.time_stepping = TimeSteppingType.UNSTEADY
    with SI_unit_system, mock_validation_context:
        output = ProbeOutput(
            name="probe",
            probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
            output_fields=["Cp"],
            output_at_final_pseudo_step_only=True,
            moving_statistic=MovingStatistic(method="mean", moving_window_size=10, start_step=100),
        )
    assert output.output_at_final_pseudo_step_only is True
    assert output.moving_statistic is not None


def test_toggle_without_moving_statistic_allowed_in_steady(mock_validation_context):
    """Toggle alone (no MovingStatistic) should be fine in steady."""
    mock_validation_context.info.time_stepping = TimeSteppingType.STEADY
    with SI_unit_system, mock_validation_context:
        output = ProbeOutput(
            name="probe",
            probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
            output_fields=["Cp"],
            output_at_final_pseudo_step_only=True,
        )
    assert output.output_at_final_pseudo_step_only is True


def test_toggle_defaults_to_false():
    output = ProbeOutput(
        name="probe",
        probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
        output_fields=["Cp"],
    )
    assert output.output_at_final_pseudo_step_only is False


def test_probe_output_accepts_toggle():
    output = ProbeOutput(
        name="probe",
        probe_points=[Point(name="pt", location=(0, 0, 0) * u.m)],
        output_fields=["Cp"],
        output_at_final_pseudo_step_only=True,
    )
    assert output.output_at_final_pseudo_step_only is True
