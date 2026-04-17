# NOTE: compute_surface_integral_unit tests have been migrated to:
#   flex/share/flow360-schema/tests/framework/expression/test_compute_surface_integral_unit.py
# Only test_process_output_field_for_integral_vector remains here (depends on client translator).

from unittest import mock

import pytest
import unyt as u
from flow360_schema.framework.expression import Expression, UserVariable

from flow360.component.simulation.services import clear_context
from flow360.component.simulation.translator.solver_translator import (
    process_output_field_for_integral,
)


class MockUnitSystem:
    def __init__(self, name, units):
        self.name = name
        self.units = units

    def __getitem__(self, key):
        return self.units[key]

    def resolve(self):
        return self


class MockParams:
    def __init__(self, unit_system_dict, unit_system_name="SI"):
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
    return MockParams({"area": 1.0 * u.m**2}, unit_system_name="SI")


def test_process_output_field_for_integral_vector(mock_params_si):
    # Test vector variable integration logic
    with mock.patch(
        "flow360.component.simulation.translator.solver_translator.solution"
    ) as mock_sol:
        with mock.patch(
            "flow360.component.simulation.translator.solver_translator.math"
        ) as mock_math:
            mock_math.magnitude.return_value = Expression(expression="1.0 * u.m**2")

            var = UserVariable(name="vec", value=[10 * u.Pa, 20 * u.Pa, 30 * u.Pa])

            processed_var = process_output_field_for_integral(var, mock_params_si)

            assert isinstance(processed_var.value, Expression)
            assert u.Unit(processed_var.value.output_units) == u.N
