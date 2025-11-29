import json
import re
import unittest

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.output_entities import Point, PointArray
from flow360.component.simulation.outputs.outputs import (
    MovingStatistic,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceProbeOutput,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.run_control.stopping_criterion import (
    StoppingCriterion,
)
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def scalar_user_variable_density():
    """A scalar UserVariable for testing."""
    return UserVariable(
        name="scalar_field",
        value=solution.density,
    )


@pytest.fixture()
def vector_user_variable_velocity():
    """A vector UserVariable for testing."""
    return UserVariable(
        name="vector_field",
        value=solution.velocity,
    )


@pytest.fixture()
def single_point_probe_output(scalar_user_variable_density, vector_user_variable_velocity):
    """A ProbeOutput with a single point."""
    return ProbeOutput(
        name="test_probe",
        output_fields=[
            scalar_user_variable_density,
            vector_user_variable_velocity,
            "mut",
            "VelocityRelative",
        ],
        probe_points=[Point(name="pt1", location=(0, 0, 0) * u.m)],
    )


@pytest.fixture()
def single_point_surface_probe_output(scalar_user_variable_density, vector_user_variable_velocity):
    """A SurfaceProbeOutput with a single point."""
    return SurfaceProbeOutput(
        name="test_surface_probe",
        output_fields=[
            scalar_user_variable_density,
            vector_user_variable_velocity,
            "mut",
            "VelocityRelative",
        ],
        probe_points=[Point(name="pt1", location=(0, 0, 0) * u.m)],
        target_surfaces=[Surface(name="wall")],
    )


@pytest.fixture()
def surface_integral_output(scalar_user_variable_density):
    """A SurfaceIntegralOutput for testing."""
    return SurfaceIntegralOutput(
        name="test_integral",
        output_fields=[scalar_user_variable_density],
        surfaces=[Surface(name="wall")],
    )


def test_criterion_scalar_field_validation(scalar_user_variable_density, single_point_probe_output):
    """Test that scalar fields are accepted."""
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.monitor_field == scalar_user_variable_density


def test_criterion_vector_field_validation_fails(
    vector_user_variable_velocity, single_point_probe_output
):
    """Test that vector fields are rejected."""
    message = "The stopping criterion can only be defined on a scalar field."
    with SI_unit_system, pytest.raises(ValueError, match=message):
        StoppingCriterion(
            monitor_field=vector_user_variable_velocity,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.m / u.s,
        )

    with SI_unit_system, pytest.raises(ValueError, match=message):
        StoppingCriterion(
            monitor_field="VelocityRelative",
            monitor_output=single_point_probe_output,
            tolerance=0.01,
        )


def test_criterion_single_point_probe_validation(
    scalar_user_variable_density, single_point_probe_output, single_point_surface_probe_output
):
    """Test that single point ProbeOutput is accepted."""
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.monitor_output == single_point_probe_output

    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_surface_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.monitor_output == single_point_surface_probe_output


def test_criterion_multi_entities_probe_validation_fails(
    scalar_user_variable_density,
    single_point_probe_output,
    single_point_surface_probe_output,
    mock_validation_context,
):
    """Test that multi-entity ProbeOutput is rejected."""
    message = (
        "For stopping criterion setup, only one single `Point` entity is allowed "
        "in `ProbeOutput`/`SurfaceProbeOutput`."
    )

    multi_point_probe_output = single_point_probe_output
    multi_point_probe_output.entities.stored_entities.append(
        Point(name="pt2", location=(1, 1, 1) * u.m)
    )
    with SI_unit_system, mock_validation_context, pytest.raises(ValueError, match=message):
        StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=multi_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )

    point_array_surface_probe_output = single_point_surface_probe_output
    point_array_surface_probe_output.entities.stored_entities = [
        PointArray(
            name="point_array",
            start=(0, 0, 0) * u.m,
            end=(1, 1, 1) * u.m,
            number_of_points=2,
        ),
    ]
    with SI_unit_system, mock_validation_context, pytest.raises(ValueError, match=message):
        StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=point_array_surface_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )


def test_criterion_field_exists_in_output_validation(single_point_probe_output):
    """Test that monitor field must exist in monitor output."""
    scalar_field = UserVariable(name="test_field", value=solution.pressure)
    message = "The monitor field does not exist in the monitor output."

    with SI_unit_system, pytest.raises(ValueError, match=message):
        criterion = StoppingCriterion(
            monitor_field=scalar_field,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.Pa,
        )


def test_criterion_field_exists_in_output_validation_success(single_point_probe_output):
    """Test successful validation when monitor field exists in output."""
    scalar_field = UserVariable(name="test_field", value=solution.pressure)
    single_point_probe_output.output_fields.append(scalar_field)
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_field,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.Pa,
        )
    assert criterion.monitor_field == scalar_field


def test_criterion_string_field_tolerance_validation(single_point_probe_output):
    """Test that string monitor fields require dimensionless tolerance."""
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field="mut",
            monitor_output=single_point_probe_output,
            tolerance=0.01,
        )
    assert criterion.tolerance == 0.01

    message = "The monitor field (mut) specified by string can only be used with a nondimensional tolerance."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        criterion = StoppingCriterion(
            monitor_field="mut",
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m / u.s,
        )


def test_criterion_dimension_matching_validation(
    scalar_user_variable_density, single_point_probe_output, surface_integral_output
):
    """Test that monitor field and tolerance dimensions must match."""
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )
    assert criterion.tolerance == 0.01 * u.kg / u.m**3

    # Valid case: surface integral tolerance's dimenision should match with field_dimensions * (length)**2
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=surface_integral_output,
            tolerance=0.01 * u.kg / u.m,
        )
    assert criterion.tolerance == 0.01 * u.kg / u.m

    # Invalid case: mismatched dimensions
    message = "The dimensions of monitor field and tolerance do not match."

    with SI_unit_system, pytest.raises(ValueError, match=message):
        StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01,  # Dimensionless tolerance for dimensional field
        )

    with SI_unit_system, pytest.raises(ValueError, match=message):
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=surface_integral_output,
            tolerance=0.01 * u.kg / u.m**3,
        )


def test_tolerance_window_size_validation(scalar_user_variable_density, single_point_probe_output):
    """Test tolerance_window_size validation."""

    # Valid case: ge=2 constraint satisfied
    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
            tolerance_window_size=5,
        )
    assert criterion.tolerance_window_size == 5

    # Invalid case: less than 2
    with SI_unit_system, pytest.raises(pd.ValidationError):
        StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
            tolerance_window_size=1,
        )


def test_criterion_with_moving_statistic(scalar_user_variable_density, single_point_probe_output):
    """Test StoppingCriterion with MovingStatistic in output."""

    single_point_probe_output.moving_statistic = MovingStatistic(
        method="deviation", moving_window_size=10
    )
    with SI_unit_system:
        criterion = StoppingCriterion(
            name="Criterion_1",
            monitor_output=single_point_probe_output,
            monitor_field=scalar_user_variable_density,
            tolerance=0.01 * u.kg / u.m**3,
        )

    assert criterion.name == "Criterion_1"
    assert criterion.monitor_output.moving_statistic.method == "deviation"
    assert criterion.monitor_output.moving_statistic.moving_window_size == 10


def test_criterion_default_values(scalar_user_variable_density, single_point_probe_output):
    """Test default values for StoppingCriterion."""

    with SI_unit_system:
        criterion = StoppingCriterion(
            monitor_field=scalar_user_variable_density,
            monitor_output=single_point_probe_output,
            tolerance=0.01 * u.kg / u.m**3,
        )

    assert criterion.name == "StoppingCriterion"
    assert criterion.tolerance_window_size is None
    assert criterion.type_name == "StoppingCriterion"


def test_criterion_with_monitor_output_id():
    # [Frontend] Simulating loading a StoppingCriterion object with the id of monitor_output,
    # ensure the validation for monitor_output works
    with open("data/simulation_stopping_criterion_webui.json", "r") as fh:
        data = json.load(fh)

    _, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 0, "monitor_output"),
            "msg": "Value error, For stopping criterion setup, only one single `Point` entity is allowed in `ProbeOutput`/`SurfaceProbeOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 1, "monitor_output"),
            "msg": "Value error, The monitor field does not exist in the monitor output.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 2, "tolerance"),
            "msg": "Value error, The dimensions of monitor field and tolerance do not match.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 3, "monitor_output"),
            "msg": "Value error, The monitor output does not exist in the outputs list.",
            "input": "1234",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]
        assert err["msg"] == exp_err["msg"]
