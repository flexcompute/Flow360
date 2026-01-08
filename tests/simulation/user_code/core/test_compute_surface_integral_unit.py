from unittest import mock

import pytest
import unyt as u

from flow360.component.simulation.services import clear_context
from flow360.component.simulation.translator.solver_translator import (
    process_output_field_for_integral,
)
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    compute_surface_integral_unit,
)


class MockUnitSystem:
    def __init__(self, name, units):
        self.name = name
        self.units = units

    def __getitem__(self, key):
        return self.units[key]


class MockParams:
    def __init__(self, unit_system_dict, unit_system_name="SI"):
        # We populate both for compatibility during transition,
        # but the test expects usage of unit_system eventually.
        self.flow360_unit_system = unit_system_dict
        self.unit_system = MockUnitSystem(unit_system_name, unit_system_dict)


@pytest.fixture(autouse=True)
def reset_context():
    """Clear user variables from the context to avoid redeclaration errors."""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def mock_params_si():
    # Simulate real behavior where UnitSystem returns unyt_quantity (1.0 * unit)
    return MockParams({"area": 1.0 * u.m**2}, unit_system_name="SI")


@pytest.fixture
def mock_params_imperial():
    return MockParams({"area": 1.0 * u.ft**2}, unit_system_name="Imperial")


def test_compute_surface_integral_unit_scalar_si(mock_params_si):
    # Variable with explicit units
    var = UserVariable(name="var", value=10 * u.Pa)
    unit = compute_surface_integral_unit(var, mock_params_si)
    # Pa * m^2 -> N
    assert u.Unit(unit) == u.N


def test_compute_surface_integral_unit_expression_si(mock_params_si):
    # Variable defined by expression
    var = UserVariable(name="var", value=Expression(expression="10 * u.Pa"))
    unit = compute_surface_integral_unit(var, mock_params_si)
    assert u.Unit(unit) == u.N


def test_compute_surface_integral_unit_dimensionless(mock_params_si):
    var = UserVariable(name="var", value=10)
    unit = compute_surface_integral_unit(var, mock_params_si)
    # dimensionless * m^2 -> m^2
    assert u.Unit(unit) == u.m**2


def test_compute_surface_integral_unit_imperial(mock_params_imperial):
    var = UserVariable(name="var", value=10 * u.lbf / u.ft**2)  # psf
    unit = compute_surface_integral_unit(var, mock_params_imperial)
    # (lbf/ft^2) * ft^2 -> lbf
    assert u.Unit(unit) == u.lbf


def test_compute_surface_integral_unit_with_output_units(mock_params_si):
    # Variable with specific output units set
    # Must use Expression to use in_units
    var = UserVariable(name="var", value=Expression(expression="10 * u.Pa")).in_units(
        new_unit="kPa"
    )
    unit = compute_surface_integral_unit(var, mock_params_si)
    # kPa * m^2 -> kN (or equivalent)
    assert u.Unit(unit) == u.kN


def test_compute_surface_integral_unit_fallback(mock_params_si):
    # Case where value is a simple number inside an expression
    var = UserVariable(name="var", value=Expression(expression="10"))
    unit = compute_surface_integral_unit(var, mock_params_si)
    assert u.Unit(unit) == u.m**2


def test_process_output_field_for_integral_vector(mock_params_si):
    # Test vector variable integration logic
    # Mock node_area_vector for this test as it's used in process_output_field_for_integral
    with mock.patch(
        "flow360.component.simulation.translator.solver_translator.solution"
    ) as mock_sol:
        # Mock magnitude call
        with mock.patch(
            "flow360.component.simulation.translator.solver_translator.math"
        ) as mock_math:
            mock_math.magnitude.return_value = Expression(expression="1.0 * u.m**2")

            # Create a vector variable
            # UserVariable automatically wraps list in Expression
            var = UserVariable(name="vec", value=[10 * u.Pa, 20 * u.Pa, 30 * u.Pa])

            # This should not raise AttributeError now
            processed_var = process_output_field_for_integral(var, mock_params_si)

            # Check if the processed variable value is an Expression
            assert isinstance(processed_var.value, Expression)
            # Ensure output_units is set on the Expression
            assert u.Unit(processed_var.value.output_units) == u.N
